"""
Parallel self-play with batched neural network inference.

Instead of playing 1 game at a time (each with N MCTS simulations done sequentially),
this module plays M games simultaneously and batches all their neural network evaluations
together into single GPU forward passes. This dramatically improves GPU utilization
from ~4% to 50-80%.

Architecture:
  - Run M games (e.g. 16-32) simultaneously
  - In each MCTS simulation step, collect all leaf states across all games
  - Batch them into one tensor and run a single GPU forward pass
  - Distribute results back to the respective MCTS trees
"""

import numpy as np
import copy
import torch
import time
from collections import deque

from game import QuoridorGame
from model import (
    QuoridorNet, encode_state, move_to_action, action_to_move,
    get_legal_action_mask, ACTION_SIZE
)


# MCTS hyperparameters (same as mcts.py)
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


class BatchMCTSNode:
    """MCTS node optimized for batch evaluation."""
    __slots__ = [
        'parent', 'action', 'prior', 'children',
        'visit_count', 'total_value', 'game_state', 'is_expanded'
    ]

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.game_state = None
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, parent_visits):
        exploration = C_PUCT * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


def _select_child(node):
    """Select child with highest UCB score."""
    best_score = -float('inf')
    best_child = None
    for child in node.children.values():
        score = child.ucb_score(node.visit_count)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _traverse_to_leaf(root):
    """
    Traverse from root to a leaf node using UCB selection.
    Returns (leaf_node, search_path).
    """
    node = root
    search_path = [node]

    while node.is_expanded and node.children:
        node = _select_child(node)
        search_path.append(node)

    # Ensure game state is available on the leaf
    if node.game_state is None and len(search_path) >= 2:
        node.game_state = copy.deepcopy(search_path[-2].game_state)
        move = action_to_move(node.action)
        node.game_state.play_move(move)

    return node, search_path


def _expand_node(node, policy):
    """Expand a node by creating children from legal moves with prior probabilities."""
    legal_mask = get_legal_action_mask(node.game_state)
    policy = policy * legal_mask
    policy_sum = policy.sum()
    if policy_sum > 0:
        policy /= policy_sum

    legal_moves = node.game_state.get_legal_moves()
    for move in legal_moves:
        action = move_to_action(move)
        child = BatchMCTSNode(parent=node, action=action, prior=policy[action])
        node.children[action] = child
    node.is_expanded = True


def _backpropagate(search_path, value):
    """Backpropagate value up the search path, alternating sign."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value
        value = -value


def _batch_evaluate(model, states_list, device):
    """
    Evaluate a batch of game states with the neural network in one forward pass.

    Args:
        model: QuoridorNet
        states_list: list of encoded state arrays (each is 12x9x9 numpy)
        device: torch device

    Returns:
        policies: list of numpy policy arrays
        values: list of float values
    """
    if not states_list:
        return [], []

    batch = torch.FloatTensor(np.array(states_list)).to(device)

    model.eval()
    with torch.no_grad():
        policy_logits, value_preds = model(batch)

    policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
    values = value_preds.squeeze(-1).cpu().numpy()

    return list(policies), list(values)


class ParallelMCTS:
    """
    Runs MCTS for multiple games simultaneously, batching all neural network
    evaluations into single forward passes for GPU efficiency.
    """

    def __init__(self, model, device='cpu', num_simulations=100, batch_size=32):
        """
        Args:
            model: QuoridorNet instance
            device: 'cpu' or 'cuda'
            num_simulations: number of MCTS simulations per move
            batch_size: number of games to run in parallel
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size

    def search_batch(self, games, temperature=1.0, add_noise=True):
        """
        Run MCTS for multiple games simultaneously with batched inference.

        Args:
            games: list of QuoridorGame instances (active games)
            temperature: exploration temperature for action selection
            add_noise: whether to add Dirichlet noise at root

        Returns:
            action_probs_list: list of action probability arrays (one per game)
        """
        num_games = len(games)
        if num_games == 0:
            return []

        # Initialize roots
        roots = []
        for game in games:
            root = BatchMCTSNode()
            root.game_state = copy.deepcopy(game)
            root.is_expanded = True
            roots.append(root)

        # Batch evaluate all roots
        root_states = [encode_state(game) for game in games]
        root_policies, root_values = _batch_evaluate(
            self.model, root_states, self.device
        )

        # Expand roots with network predictions and optional noise
        for i, (root, policy) in enumerate(zip(roots, root_policies)):
            legal_mask = get_legal_action_mask(root.game_state)
            policy = policy * legal_mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy /= policy_sum

            if add_noise:
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * ACTION_SIZE)
                policy = (1 - DIRICHLET_EPSILON) * policy + DIRICHLET_EPSILON * noise
                policy = policy * legal_mask
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy /= policy_sum

            legal_moves = root.game_state.get_legal_moves()
            for move in legal_moves:
                action = move_to_action(move)
                child = BatchMCTSNode(parent=root, action=action, prior=policy[action])
                root.children[action] = child

        # Run simulations with batched evaluation
        for sim in range(self.num_simulations):
            # Phase 1: Traverse to leaves (CPU-bound, fast)
            leaves = []
            search_paths = []
            leaf_states = []
            terminal_indices = []  # indices where game ended (no NN eval needed)
            eval_indices = []  # indices that need NN evaluation

            for game_idx in range(num_games):
                leaf, search_path = _traverse_to_leaf(roots[game_idx])
                leaves.append(leaf)
                search_paths.append(search_path)

                # Check terminal state
                winner = leaf.game_state.get_winner()
                if winner != 0:
                    terminal_indices.append(game_idx)
                else:
                    eval_indices.append(game_idx)
                    leaf_states.append(encode_state(leaf.game_state))

            # Phase 2: Batch evaluate all non-terminal leaves (GPU, one pass)
            if eval_indices:
                policies, values = _batch_evaluate(
                    self.model, leaf_states, self.device
                )
            else:
                policies, values = [], []

            # Phase 3: Expand and backpropagate
            eval_ptr = 0
            for game_idx in range(num_games):
                leaf = leaves[game_idx]
                search_path = search_paths[game_idx]

                if game_idx in terminal_indices:
                    # Terminal node: value from parent's perspective
                    winner = leaf.game_state.get_winner()
                    parent_player = search_path[-2].game_state.current_player
                    value = 1.0 if winner == parent_player else -1.0
                else:
                    # Non-terminal: expand with network output
                    policy = policies[eval_ptr]
                    value = values[eval_ptr]
                    eval_ptr += 1

                    _expand_node(leaf, policy)
                    # Value is from current player's perspective, negate for backprop
                    value = -float(value)

                _backpropagate(search_path, value)

        # Compute action probabilities from visit counts
        action_probs_list = []
        for root in roots:
            action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            for action, child in root.children.items():
                action_probs[action] = child.visit_count

            if temperature == 0:
                best = np.argmax(action_probs)
                action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
                action_probs[best] = 1.0
            else:
                if action_probs.sum() > 0:
                    action_probs = action_probs ** (1.0 / temperature)
                    action_probs /= action_probs.sum()

            action_probs_list.append(action_probs)

        return action_probs_list


def parallel_self_play_games(model, device='cpu', num_games=16, num_simulations=100,
                             temp_threshold=15, record_moves=False):
    """
    Play multiple self-play games simultaneously with batched inference.

    This is the key function for GPU utilization. Instead of playing games
    sequentially (each making individual NN calls), we play N games at once
    and batch all evaluations.

    Args:
        model: QuoridorNet instance
        device: 'cpu' or 'cuda'
        num_games: number of games to play in parallel
        num_simulations: MCTS simulations per move
        temp_threshold: move number after which temperature drops
        record_moves: whether to record move sequences for replay

    Returns:
        all_examples: list of (state, policy, value) training tuples
        game_replays: list of replay dicts (if record_moves=True, else empty list)
    """
    # Initialize all games
    games = [QuoridorGame() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]  # (state, pi, current_player)
    move_sequences = [[] for _ in range(num_games)]
    move_counts = [0] * num_games
    finished = [False] * num_games
    winners = [0] * num_games

    pmcts = ParallelMCTS(model, device=device, num_simulations=num_simulations,
                         batch_size=num_games)

    while not all(finished):
        # Gather active (unfinished) games
        active_indices = [i for i in range(num_games) if not finished[i]]
        active_games = [games[i] for i in active_indices]

        # Determine temperatures for active games
        temperatures = []
        for i in active_indices:
            t = 1.0 if move_counts[i] < temp_threshold else 0.1
            temperatures.append(t)

        # We batch all active games at the same temperature bucket
        # For simplicity, use max temperature (slight approximation when games
        # are at different stages - this is fine since the difference is small
        # and both explore/exploit correctly via their own tree statistics)
        # Actually, let's be precise: group by temperature
        temp_groups = {}
        for idx, (game_idx, temp) in enumerate(zip(active_indices, temperatures)):
            temp_key = round(temp, 2)
            if temp_key not in temp_groups:
                temp_groups[temp_key] = []
            temp_groups[temp_key].append((idx, game_idx))

        # Run batched MCTS for each temperature group
        all_action_probs = [None] * len(active_indices)

        for temp_val, group in temp_groups.items():
            group_games = [active_games[local_idx] for local_idx, _ in group]
            probs = pmcts.search_batch(group_games, temperature=temp_val, add_noise=True)
            for (local_idx, _), pi in zip(group, probs):
                all_action_probs[local_idx] = pi

        # Apply moves for each active game
        for local_idx, game_idx in enumerate(active_indices):
            pi = all_action_probs[local_idx]

            # Store training example
            state = encode_state(games[game_idx])
            histories[game_idx].append((state, pi, games[game_idx].current_player))

            # Sample action
            action = np.random.choice(ACTION_SIZE, p=pi)
            move = action_to_move(action)

            # Record move if requested
            if record_moves:
                from main import move_to_notation
                move_sequences[game_idx].append({
                    'player': games[game_idx].current_player,
                    'notation': move_to_notation(move),
                    'move_tuple': list(move),
                })

            # Play move
            games[game_idx].play_move(move)
            move_counts[game_idx] += 1

            # Check termination
            winner = games[game_idx].get_winner()
            if winner != 0 or move_counts[game_idx] > 200:
                finished[game_idx] = True
                winners[game_idx] = winner

    # Compile training examples with outcome labels
    all_examples = []
    game_replays = []

    for game_idx in range(num_games):
        winner = winners[game_idx]
        for state, pi, player in histories[game_idx]:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            all_examples.append((state, pi, value))

        if record_moves:
            game_replays.append({
                'moves': move_sequences[game_idx],
                'winner': winner,
                'length': move_counts[game_idx],
            })

    return all_examples, game_replays


def generate_self_play_data_parallel(model, device='cpu', num_games=100,
                                     num_simulations=100, batch_size=16,
                                     record_replays=False):
    """
    Generate training data using parallel self-play with batched inference.

    Drop-in replacement for self_play.generate_self_play_data() but with
    dramatically better GPU utilization.

    Args:
        model: QuoridorNet instance
        device: 'cpu' or 'cuda'
        num_games: total number of games to play
        num_simulations: MCTS simulations per move
        batch_size: number of games to play simultaneously
        record_replays: whether to record move sequences

    Returns:
        Same as generate_self_play_data:
        If record_replays is False:
            Tuple of (examples, avg_game_length)
        If record_replays is True:
            Tuple of (examples, avg_game_length, game_replays)
    """
    all_examples = []
    all_replays = []
    game_lengths = []

    # Process games in batches
    games_remaining = num_games
    games_completed = 0

    while games_remaining > 0:
        current_batch = min(batch_size, games_remaining)

        examples, replays = parallel_self_play_games(
            model, device=device,
            num_games=current_batch,
            num_simulations=num_simulations,
            record_moves=record_replays
        )

        all_examples.extend(examples)
        if record_replays:
            all_replays.extend(replays)
            for replay in replays:
                game_lengths.append(replay['length'])
        else:
            # Estimate game lengths from examples count / batch size
            # Each game contributes move_count examples
            avg_this_batch = len(examples) / current_batch if current_batch > 0 else 0
            game_lengths.extend([avg_this_batch] * current_batch)

        games_remaining -= current_batch
        games_completed += current_batch

        if games_completed % max(1, batch_size) == 0 or games_remaining == 0:
            print(f"  Parallel self-play: {games_completed}/{num_games} games, "
                  f"{len(all_examples)} positions")

    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    print(f"  Average game length: {avg_game_length:.1f} moves")

    if record_replays:
        return all_examples, avg_game_length, all_replays

    return all_examples, avg_game_length

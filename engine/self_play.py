"""
Self-play: the neural network plays against itself to generate training data.

Each game produces a list of (state, policy, value) tuples:
  - state: encoded board position
  - policy: MCTS visit-count distribution (target for policy head)
  - value: eventual game result from this player's perspective (target for value head)
"""

import math
import numpy as np
import copy
from collections import deque
from game import QuoridorGame
from model import encode_state, action_to_move, ACTION_SIZE
from mcts import MCTS

# Default training hyperparameters
DEFAULT_GAMMA = 0.98            # Discount factor for outcome rewards
DEFAULT_MAX_GAME_MOVES = 60     # Truncated game cap
DEFAULT_ASYMMETRIC_RATIO = 0.2  # Fraction of games with asymmetric walls


def _bfs_distance(game, start_pos, target_row):
    """BFS shortest path distance from start_pos to target_row, considering walls."""
    q = deque([(start_pos, 0)])
    visited = set([start_pos])

    while q:
        (r, c), dist = q.popleft()
        if r == target_row:
            return dist

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                if not game._is_blocked(r, c, nr, nc):
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))

    return 50  # unreachable (shouldn't happen with valid game state)

# Try to import C++ backend for 100-1000x speedup
try:
    import quoridor_cpp
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False


def _apply_asymmetric_walls(game, asymmetric_ratio):
    """With asymmetric_ratio probability, give one player 10 walls and the other 0."""
    if np.random.random() < asymmetric_ratio:
        if np.random.random() < 0.5:
            game.p1_walls = 10
            game.p2_walls = 0
        else:
            game.p1_walls = 0
            game.p2_walls = 10
        return True
    return False


def _compute_discounted_values(history, winner, game, gamma, max_game_moves):
    """
    Compute discounted outcome rewards for training examples.

    For wins/losses: V_t = z * gamma^(N-1-t)
    For truncated games (no winner): uses BFS-based relative score with discount.
    """
    N = len(history)
    training_examples = []

    if winner == 0:
        # Truncated game — compute relative BFS score
        p1_dist = _bfs_distance(game, game.p1_pos, 8)
        p2_dist = _bfs_distance(game, game.p2_pos, 0)
        v_cap = math.tanh((p2_dist - p1_dist) / 4.0)  # positive if P1 is closer

        for t, (state, pi, player) in enumerate(history):
            # V_cap from this player's perspective
            v_player = v_cap if player == 1 else -v_cap
            # Apply discount
            value = v_player * (gamma ** (N - 1 - t))
            training_examples.append((state, pi, value))
    else:
        # Clear winner — discounted reward
        for t, (state, pi, player) in enumerate(history):
            z = 1.0 if winner == player else -1.0
            value = z * (gamma ** (N - 1 - t))
            training_examples.append((state, pi, value))

    return training_examples


def self_play_game(model, device='cpu', num_simulations=100, temp_threshold=12,
                   record_moves=False, gamma=DEFAULT_GAMMA,
                   max_game_moves=DEFAULT_MAX_GAME_MOVES,
                   asymmetric_ratio=DEFAULT_ASYMMETRIC_RATIO):
    """
    Play one full game of self-play.

    Args:
        model: the neural network
        device: 'cpu' or 'cuda'
        num_simulations: MCTS simulations per move
        temp_threshold: move number after which temperature drops to 0
        record_moves: if True, also return the move sequence as a replay dict
        gamma: discount factor for outcome rewards (default 0.98)
        max_game_moves: truncate game after this many moves (default 60)
        asymmetric_ratio: fraction of games with asymmetric walls (default 0.2)

    Returns:
        If record_moves is False:
            List of (state, policy, value) training examples
        If record_moves is True:
            Tuple of (training_examples, replay_dict) where replay_dict has:
              moves: [{player, notation, move_tuple}], winner, length
    """
    game = QuoridorGame()

    # Asymmetric wall curriculum: 20% of games have one player with 0 walls
    _apply_asymmetric_walls(game, asymmetric_ratio)

    mcts = MCTS(model, device=device, num_simulations=num_simulations)

    history = []  # (encoded_state, pi, current_player)
    move_sequence = []  # recorded moves for replay
    move_count = 0

    while game.get_winner() == 0:
        # Temperature: explore early (1.5 for more randomness), exploit later
        temperature = 1.5 if move_count < temp_threshold else 0.1

        # Run MCTS
        pi = mcts.search(game, temperature=temperature, add_noise=True)

        # Store training example
        state = encode_state(game)
        history.append((state, pi, game.current_player))

        # Sample action from pi
        action = np.random.choice(ACTION_SIZE, p=pi)
        move = action_to_move(action)

        # Record the move if requested
        if record_moves:
            from main import move_to_notation
            move_sequence.append({
                'player': game.current_player,
                'notation': move_to_notation(move),
                'move_tuple': list(move),  # Convert tuple to list for JSON
            })

        game.play_move(move)
        move_count += 1

        # Truncated game cap
        if move_count >= max_game_moves:
            break

    # Assign discounted values based on game outcome
    winner = game.get_winner()
    training_examples = _compute_discounted_values(history, winner, game, gamma, max_game_moves)

    if record_moves:
        replay_dict = {
            'moves': move_sequence,
            'winner': winner,
            'length': move_count,
        }
        return training_examples, replay_dict

    return training_examples


def _bfs_distance_cpp(game, start_pos, target_row):
    """BFS shortest path distance for C++ game objects."""
    q = deque([(start_pos, 0)])
    visited = set([start_pos])

    while q:
        (r, c), dist = q.popleft()
        if r == target_row:
            return dist

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                if not game.is_blocked(r, c, nr, nc):
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))

    return 50  # unreachable


def self_play_game_cpp(model, device='cpu', num_simulations=100, temp_threshold=12,
                       batch_size=8, gamma=DEFAULT_GAMMA,
                       max_game_moves=DEFAULT_MAX_GAME_MOVES,
                       asymmetric_ratio=DEFAULT_ASYMMETRIC_RATIO,
                       force_wall=False):
    """
    Play one full game of self-play using the C++ backend.
    100-1000x faster than the Python version due to:
      - C++ game state copy (~200 bytes memcpy vs Python deepcopy ~1ms)
      - No Python overhead in MCTS tree traversal
      - Batched neural network evaluation (fewer Python<->C++ roundtrips)

    Args:
        force_wall: if True, force a random filtered wall move on a randomly chosen
                    early turn (2, 4, or 6) to ensure wall trajectories in replay buffer.

    Returns:
        List of (state, policy, value) training examples
    """
    import torch

    game = quoridor_cpp.QuoridorGame()

    # Asymmetric wall curriculum: randomly give one player 0 walls
    if np.random.random() < asymmetric_ratio:
        if np.random.random() < 0.5:
            game.p1_walls = 10
            game.p2_walls = 0
        else:
            game.p1_walls = 0
            game.p2_walls = 10

    # For forced wall injection: pick one turn from {2, 4, 6} to force a wall
    forced_wall_turn = None
    if force_wall:
        forced_wall_turn = np.random.choice([2, 4, 6])

    # Create the batch evaluation function for the neural network
    def evaluate_batch(states_numpy):
        """
        Evaluate a batch of game states with the neural network.
        states_numpy: numpy array of shape (batch_size, 12, 9, 9)
        Returns: (policies, values) as numpy arrays
        """
        tensor = torch.FloatTensor(states_numpy).to(device)
        model.eval()
        with torch.no_grad():
            policy_logits, values = model(tensor)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()
        return policies, values

    history = []  # (encoded_state, pi, current_player)
    move_count = 0

    while game.get_winner() == 0:
        # Temperature: explore early (1.5 for more randomness), exploit later
        temperature = 1.5 if move_count < temp_threshold else 0.1

        # Forced wall injection: on the chosen turn, skip MCTS and play a random wall
        if force_wall and move_count == forced_wall_turn:
            filtered_moves = game.get_filtered_legal_actions()
            wall_moves = [m for m in filtered_moves if m.type != quoridor_cpp.MoveType.PAWN]

            if wall_moves:
                # Pick a random wall from filtered candidates
                chosen_wall = wall_moves[np.random.randint(len(wall_moves))]
                action = chosen_wall.to_action()

                # Create a policy that puts weight on this action
                pi = np.zeros(ACTION_SIZE, dtype=np.float32)
                pi[action] = 1.0

                # Store training example
                state = np.array(quoridor_cpp.encode_state(game))
                history.append((state, pi, game.current_player))

                game.play_move(chosen_wall)
                move_count += 1
                if move_count >= max_game_moves:
                    break
                continue

        # Run C++ MCTS with batched evaluation
        pi = quoridor_cpp.mcts_search(
            game,
            num_sims=num_simulations,
            batch_size=batch_size,
            temperature=temperature,
            add_noise=True,
            eval_fn=evaluate_batch
        )

        # Store training example (encode state in C++ too)
        state = np.array(quoridor_cpp.encode_state(game))
        history.append((state, pi, game.current_player))

        # Sample action from pi
        action = np.random.choice(ACTION_SIZE, p=pi)
        move = quoridor_cpp.Move.from_action(action)
        game.play_move(move)
        move_count += 1

        # Truncated game cap
        if move_count >= max_game_moves:
            break

    # Assign discounted values based on game outcome
    winner = game.get_winner()
    N = len(history)
    training_examples = []

    if winner == 0:
        # Truncated game — compute relative BFS score
        p1_pos = game.p1_pos  # tuple (row, col)
        p2_pos = game.p2_pos
        p1_dist = _bfs_distance_cpp(game, p1_pos, 8)
        p2_dist = _bfs_distance_cpp(game, p2_pos, 0)
        v_cap = math.tanh((p2_dist - p1_dist) / 4.0)  # positive if P1 is closer

        for t, (state, pi, player) in enumerate(history):
            v_player = v_cap if player == 1 else -v_cap
            value = v_player * (gamma ** (N - 1 - t))
            training_examples.append((state, pi, value))
    else:
        for t, (state, pi, player) in enumerate(history):
            z = 1.0 if winner == player else -1.0
            value = z * (gamma ** (N - 1 - t))
            training_examples.append((state, pi, value))

    return training_examples


def _wall_bot_move(game):
    """
    Wall bot heuristic: places walls on opponent's shortest path for the first 3 walls,
    then rushes forward. Uses BFS to find the opponent's path and block it.

    Returns: a Move object (quoridor_cpp.Move) to play.
    """
    current = game.current_player
    my_pos = game.p1_pos if current == 1 else game.p2_pos
    my_target = 8 if current == 1 else 0
    opp_pos = game.p2_pos if current == 1 else game.p1_pos
    opp_target = 0 if current == 1 else 8

    walls_placed = 10 - (game.p1_walls if current == 1 else game.p2_walls)

    # Phase 1: place walls on opponent's path (first 3 walls)
    if walls_placed < 3:
        walls_left = game.p1_walls if current == 1 else game.p2_walls
        if walls_left > 0:
            # Get filtered wall moves and pick the one that increases opponent distance the most
            filtered_moves = game.get_filtered_legal_actions()
            wall_moves = [m for m in filtered_moves if m.type != quoridor_cpp.MoveType.PAWN]

            if wall_moves:
                # Evaluate each wall by how much it increases opponent's distance
                best_wall = None
                best_increase = 0
                current_opp_dist = _bfs_distance_cpp(game, opp_pos, opp_target)

                for wall in wall_moves:
                    test_game = quoridor_cpp.QuoridorGame(game)
                    test_game.play_move(wall)
                    new_dist = _bfs_distance_cpp(test_game, opp_pos, opp_target)
                    increase = new_dist - current_opp_dist
                    if increase > best_increase:
                        best_increase = increase
                        best_wall = wall

                if best_wall is not None:
                    return best_wall

    # Phase 2: rush forward (pick the pawn move closest to goal)
    pawn_moves = game.get_legal_pawn_moves()
    if pawn_moves:
        best_pawn = None
        best_dist = 50

        for move in pawn_moves:
            # Simulate the pawn move and check BFS distance
            test_game = quoridor_cpp.QuoridorGame(game)
            test_game.play_move(move)
            # After move, current_player switched, so check from new position
            new_pos = test_game.p1_pos if current == 1 else test_game.p2_pos
            dist = _bfs_distance_cpp(test_game, new_pos, my_target)
            if dist < best_dist:
                best_dist = dist
                best_pawn = move

        if best_pawn is not None:
            return best_pawn

    # Fallback: play any legal move
    legal_moves = game.get_legal_moves()
    if legal_moves:
        return legal_moves[0]
    return None


def self_play_game_vs_opponent(model, device='cpu', num_simulations=100,
                               temp_threshold=12, batch_size=8,
                               gamma=DEFAULT_GAMMA, max_game_moves=DEFAULT_MAX_GAME_MOVES,
                               opponent_type='wall_bot', opponent_model=None):
    """
    Play a self-play game where the current model plays against a specific opponent.
    The model alternates randomly between playing as P1 and P2.

    Args:
        opponent_type: 'wall_bot' or 'old_model'
        opponent_model: required if opponent_type is 'old_model'

    Returns:
        List of (state, policy, value) training examples (only from model's perspective)
    """
    import torch

    game = quoridor_cpp.QuoridorGame()

    # Randomly assign which player the model controls
    model_player = np.random.choice([1, 2])

    def evaluate_batch(states_numpy):
        tensor = torch.FloatTensor(states_numpy).to(device)
        model.eval()
        with torch.no_grad():
            policy_logits, values = model(tensor)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()
        return policies, values

    def evaluate_batch_opponent(states_numpy):
        tensor = torch.FloatTensor(states_numpy).to(device)
        opponent_model.eval()
        with torch.no_grad():
            policy_logits, values = opponent_model(tensor)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(-1).cpu().numpy()
        return policies, values

    history = []  # (encoded_state, pi, current_player) - only model's moves
    move_count = 0

    while game.get_winner() == 0:
        is_model_turn = (game.current_player == model_player)

        if is_model_turn:
            # Model's turn: use MCTS
            temperature = 1.5 if move_count < temp_threshold else 0.1
            pi = quoridor_cpp.mcts_search(
                game,
                num_sims=num_simulations,
                batch_size=batch_size,
                temperature=temperature,
                add_noise=True,
                eval_fn=evaluate_batch
            )

            state = np.array(quoridor_cpp.encode_state(game))
            history.append((state, pi, game.current_player))

            action = np.random.choice(ACTION_SIZE, p=pi)
            move = quoridor_cpp.Move.from_action(action)
        else:
            # Opponent's turn
            if opponent_type == 'wall_bot':
                move = _wall_bot_move(game)
            elif opponent_type == 'old_model' and opponent_model is not None:
                pi = quoridor_cpp.mcts_search(
                    game,
                    num_sims=num_simulations // 2,  # opponent uses fewer sims
                    batch_size=batch_size,
                    temperature=0.5,
                    add_noise=False,
                    eval_fn=evaluate_batch_opponent
                )
                action = np.random.choice(ACTION_SIZE, p=pi)
                move = quoridor_cpp.Move.from_action(action)
            else:
                # Fallback: random legal move
                legal = game.get_legal_moves()
                move = legal[np.random.randint(len(legal))]

        if move is not None:
            game.play_move(move)
        move_count += 1

        if move_count >= max_game_moves:
            break

    # Assign values only to model's training examples
    winner = game.get_winner()
    N = len(history)
    training_examples = []

    if winner == 0:
        p1_pos = game.p1_pos
        p2_pos = game.p2_pos
        p1_dist = _bfs_distance_cpp(game, p1_pos, 8)
        p2_dist = _bfs_distance_cpp(game, p2_pos, 0)
        v_cap = math.tanh((p2_dist - p1_dist) / 4.0)

        for t, (state, pi, player) in enumerate(history):
            v_player = v_cap if player == 1 else -v_cap
            value = v_player * (gamma ** (N - 1 - t))
            training_examples.append((state, pi, value))
    else:
        for t, (state, pi, player) in enumerate(history):
            z = 1.0 if winner == player else -1.0
            value = z * (gamma ** (N - 1 - t))
            training_examples.append((state, pi, value))

    return training_examples


def generate_self_play_data(model, device='cpu', num_games=100, num_simulations=100,
                            record_replays=False, parallel=None, batch_size=16,
                            gamma=DEFAULT_GAMMA, max_game_moves=DEFAULT_MAX_GAME_MOVES,
                            asymmetric_ratio=DEFAULT_ASYMMETRIC_RATIO,
                            opponent_pool=True, old_model=None):
    """
    Generate training data from multiple self-play games.

    Args:
        model: the neural network
        device: 'cpu' or 'cuda'
        num_games: number of games to play
        num_simulations: MCTS simulations per move
        record_replays: if True, also record and return full game replays
        parallel: if True, use batched parallel self-play for better GPU utilization.
                  If None, auto-detect (use parallel on CUDA devices).
        batch_size: number of games to play simultaneously when parallel=True
        gamma: discount factor for outcome rewards (default 0.98)
        max_game_moves: truncate game after this many moves (default 60)
        asymmetric_ratio: fraction of games with asymmetric walls (default 0.2)
        opponent_pool: if True, split games into 70% self-play, 20% vs old model, 10% vs wall bot
        old_model: model from ~5 iterations ago for opponent pool (None = skip old model games)

    Returns:
        If record_replays is False:
            Tuple of (examples, avg_game_length)
        If record_replays is True:
            Tuple of (examples, avg_game_length, game_replays) where
            game_replays is a list of replay dicts
    """
    # Use C++ backend if available (massive speedup)
    if HAS_CPP_BACKEND and not record_replays:
        print(f"  Using C++ backend (quoridor_cpp) for {num_games} games")
        all_examples = []
        game_lengths = []
        cpp_batch_size = min(batch_size, 32)  # batch size for NN eval within MCTS

        # Split games into opponent pool categories
        if opponent_pool:
            n_self_play = int(num_games * 0.70)
            n_old_model = int(num_games * 0.20) if old_model is not None else 0
            n_wall_bot = num_games - n_self_play - n_old_model
            # If no old model available, give those games to self-play
            if old_model is None:
                n_self_play += int(num_games * 0.20)
            print(f"  Opponent pool: {n_self_play} self-play, {n_old_model} vs old model, {n_wall_bot} vs wall bot")
        else:
            n_self_play = num_games
            n_old_model = 0
            n_wall_bot = 0

        # Forced wall injection: 20% of self-play games
        n_forced_wall = int(n_self_play * 0.20)

        game_idx = 0

        # --- Self-play games ---
        for i in range(n_self_play):
            force_wall = (i < n_forced_wall)
            examples = self_play_game_cpp(
                model, device=device, num_simulations=num_simulations,
                batch_size=cpp_batch_size, gamma=gamma,
                max_game_moves=max_game_moves,
                asymmetric_ratio=asymmetric_ratio,
                force_wall=force_wall
            )
            game_lengths.append(len(examples))
            all_examples.extend(examples)
            game_idx += 1
            if game_idx % 10 == 0:
                print(f"  Self-play (C++): {game_idx}/{num_games} games, {len(all_examples)} positions")

        # --- Games vs old model ---
        for i in range(n_old_model):
            examples = self_play_game_vs_opponent(
                model, device=device, num_simulations=num_simulations,
                batch_size=cpp_batch_size, gamma=gamma,
                max_game_moves=max_game_moves,
                opponent_type='old_model', opponent_model=old_model
            )
            game_lengths.append(len(examples))
            all_examples.extend(examples)
            game_idx += 1
            if game_idx % 10 == 0:
                print(f"  Self-play (C++): {game_idx}/{num_games} games, {len(all_examples)} positions")

        # --- Games vs wall bot ---
        for i in range(n_wall_bot):
            examples = self_play_game_vs_opponent(
                model, device=device, num_simulations=num_simulations,
                batch_size=cpp_batch_size, gamma=gamma,
                max_game_moves=max_game_moves,
                opponent_type='wall_bot'
            )
            game_lengths.append(len(examples))
            all_examples.extend(examples)
            game_idx += 1
            if game_idx % 10 == 0:
                print(f"  Self-play (C++): {game_idx}/{num_games} games, {len(all_examples)} positions")

        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        print(f"  Average game length: {avg_game_length:.1f} moves")
        return all_examples, avg_game_length

    # Auto-detect: use parallel on GPU, sequential on CPU
    if parallel is None:
        parallel = (device != 'cpu')

    if parallel:
        from parallel_mcts import generate_self_play_data_parallel
        print(f"  Using parallel self-play (batch_size={batch_size}, device={device})")
        return generate_self_play_data_parallel(
            model, device=device, num_games=num_games,
            num_simulations=num_simulations, batch_size=batch_size,
            record_replays=record_replays
        )

    # Original sequential self-play (fallback)
    all_examples = []
    game_lengths = []
    game_replays = []

    for i in range(num_games):
        if record_replays:
            examples, replay = self_play_game(
                model, device=device, num_simulations=num_simulations,
                record_moves=True, gamma=gamma,
                max_game_moves=max_game_moves,
                asymmetric_ratio=asymmetric_ratio
            )
            game_replays.append(replay)
        else:
            examples = self_play_game(
                model, device=device, num_simulations=num_simulations,
                gamma=gamma, max_game_moves=max_game_moves,
                asymmetric_ratio=asymmetric_ratio
            )
        game_lengths.append(len(examples))
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Self-play: {i+1}/{num_games} games, {len(all_examples)} positions")

    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    print(f"  Average game length: {avg_game_length:.1f} moves")

    if record_replays:
        return all_examples, avg_game_length, game_replays

    return all_examples, avg_game_length

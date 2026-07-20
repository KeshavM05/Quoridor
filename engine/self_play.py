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


def self_play_game(model, device='cpu', num_simulations=100, temp_threshold=15,
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
        # Temperature: explore early, exploit later
        temperature = 1.0 if move_count < temp_threshold else 0.1

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


def self_play_game_cpp(model, device='cpu', num_simulations=100, temp_threshold=15,
                       batch_size=8, gamma=DEFAULT_GAMMA,
                       max_game_moves=DEFAULT_MAX_GAME_MOVES,
                       asymmetric_ratio=DEFAULT_ASYMMETRIC_RATIO):
    """
    Play one full game of self-play using the C++ backend.
    100-1000x faster than the Python version due to:
      - C++ game state copy (~200 bytes memcpy vs Python deepcopy ~1ms)
      - No Python overhead in MCTS tree traversal
      - Batched neural network evaluation (fewer Python<->C++ roundtrips)

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
        # Temperature: explore early, exploit later
        temperature = 1.0 if move_count < temp_threshold else 0.1

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


def generate_self_play_data(model, device='cpu', num_games=100, num_simulations=100,
                            record_replays=False, parallel=None, batch_size=16,
                            gamma=DEFAULT_GAMMA, max_game_moves=DEFAULT_MAX_GAME_MOVES,
                            asymmetric_ratio=DEFAULT_ASYMMETRIC_RATIO):
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

        for i in range(num_games):
            examples = self_play_game_cpp(
                model, device=device, num_simulations=num_simulations,
                batch_size=cpp_batch_size, gamma=gamma,
                max_game_moves=max_game_moves,
                asymmetric_ratio=asymmetric_ratio
            )
            game_lengths.append(len(examples))
            all_examples.extend(examples)
            if (i + 1) % 10 == 0:
                print(f"  Self-play (C++): {i+1}/{num_games} games, {len(all_examples)} positions")

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

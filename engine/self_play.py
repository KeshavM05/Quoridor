"""
Self-play: the neural network plays against itself to generate training data.

Each game produces a list of (state, policy, value) tuples:
  - state: encoded board position
  - policy: MCTS visit-count distribution (target for policy head)
  - value: eventual game result from this player's perspective (target for value head)
"""

import numpy as np
import copy
from game import QuoridorGame
from model import encode_state, action_to_move, ACTION_SIZE
from mcts import MCTS

# Try to import C++ backend for 100-1000x speedup
try:
    import quoridor_cpp
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False


def self_play_game(model, device='cpu', num_simulations=100, temp_threshold=15,
                   record_moves=False):
    """
    Play one full game of self-play.

    Args:
        model: the neural network
        device: 'cpu' or 'cuda'
        num_simulations: MCTS simulations per move
        temp_threshold: move number after which temperature drops to 0
        record_moves: if True, also return the move sequence as a replay dict

    Returns:
        If record_moves is False:
            List of (state, policy, value) training examples
        If record_moves is True:
            Tuple of (training_examples, replay_dict) where replay_dict has:
              moves: [{player, notation, move_tuple}], winner, length
    """
    game = QuoridorGame()
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

        # Safety: cap game length (shorter = faster training iterations)
        if move_count > 100:
            break

    # Assign values based on game outcome
    winner = game.get_winner()
    training_examples = []

    for state, pi, player in history:
        if winner == 0:
            value = 0.0  # Draw (timeout)
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_examples.append((state, pi, value))

    if record_moves:
        replay_dict = {
            'moves': move_sequence,
            'winner': winner,
            'length': move_count,
        }
        return training_examples, replay_dict

    return training_examples


def self_play_game_cpp(model, device='cpu', num_simulations=100, temp_threshold=15,
                       batch_size=8):
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

        # Safety: cap game length
        if move_count > 100:
            break

    # Assign values based on game outcome
    winner = game.get_winner()
    training_examples = []

    for state, pi, player in history:
        if winner == 0:
            value = 0.0  # Draw (timeout)
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_examples.append((state, pi, value))

    return training_examples


def generate_self_play_data(model, device='cpu', num_games=100, num_simulations=100,
                            record_replays=False, parallel=None, batch_size=16):
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
                batch_size=cpp_batch_size
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
                record_moves=True
            )
            game_replays.append(replay)
        else:
            examples = self_play_game(
                model, device=device, num_simulations=num_simulations
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

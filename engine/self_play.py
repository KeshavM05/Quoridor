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


def self_play_game(model, device='cpu', num_simulations=100, temp_threshold=15):
    """
    Play one full game of self-play.

    Args:
        model: the neural network
        device: 'cpu' or 'cuda'
        num_simulations: MCTS simulations per move
        temp_threshold: move number after which temperature drops to 0

    Returns:
        List of (state, policy, value) training examples
    """
    game = QuoridorGame()
    mcts = MCTS(model, device=device, num_simulations=num_simulations)

    history = []  # (encoded_state, pi, current_player)
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
        game.play_move(move)
        move_count += 1

        # Safety: cap game length
        if move_count > 200:
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


def generate_self_play_data(model, device='cpu', num_games=100, num_simulations=100):
    """
    Generate training data from multiple self-play games.

    Returns:
        Tuple of (examples, avg_game_length) where:
          - examples: List of (state, policy, value) tuples
          - avg_game_length: average number of moves per game
    """
    all_examples = []
    game_lengths = []

    for i in range(num_games):
        examples = self_play_game(model, device=device, num_simulations=num_simulations)
        game_lengths.append(len(examples))
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Self-play: {i+1}/{num_games} games, {len(all_examples)} positions")

    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    print(f"  Average game length: {avg_game_length:.1f} moves")

    return all_examples, avg_game_length

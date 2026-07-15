"""
Arena: pit two models against each other to determine which is stronger.
"""

import numpy as np
import copy
from game import QuoridorGame
from model import action_to_move, ACTION_SIZE
from mcts import MCTS


def play_one_game(model1, model2, device='cpu', num_simulations=50):
    """
    Play one game between model1 (player 1) and model2 (player 2).
    Returns: 1 if model1 wins, 2 if model2 wins, 0 if draw/timeout.
    """
    game = QuoridorGame()
    mcts1 = MCTS(model1, device=device, num_simulations=num_simulations)
    mcts2 = MCTS(model2, device=device, num_simulations=num_simulations)

    move_count = 0
    while game.get_winner() == 0:
        if game.current_player == 1:
            pi = mcts1.search(game, temperature=0.1, add_noise=False)
        else:
            pi = mcts2.search(game, temperature=0.1, add_noise=False)

        action = np.argmax(pi)
        move = action_to_move(action)

        if move not in game.get_legal_moves():
            # Fallback: pick best legal action
            legal = game.get_legal_moves()
            if not legal:
                break
            from model import move_to_action
            legal_actions = [move_to_action(m) for m in legal]
            best_idx = np.argmax([pi[a] for a in legal_actions])
            move = legal[best_idx]

        game.play_move(move)
        move_count += 1

        if move_count > 200:
            return 0

    return game.get_winner()


def pit_models(new_model, old_model, device='cpu', num_games=40, num_simulations=50):
    """
    Play num_games between new_model and old_model, alternating who goes first.

    Returns: (new_wins, old_wins, draws)
    """
    new_wins = 0
    old_wins = 0
    draws = 0

    for i in range(num_games):
        if i % 2 == 0:
            # New model plays as player 1
            result = play_one_game(new_model, old_model, device=device, num_simulations=num_simulations)
            if result == 1:
                new_wins += 1
            elif result == 2:
                old_wins += 1
            else:
                draws += 1
        else:
            # New model plays as player 2
            result = play_one_game(old_model, new_model, device=device, num_simulations=num_simulations)
            if result == 2:
                new_wins += 1
            elif result == 1:
                old_wins += 1
            else:
                draws += 1

        if (i + 1) % 10 == 0:
            print(f"    Arena: {i+1}/{num_games} — new: {new_wins}, old: {old_wins}, draws: {draws}")

    return new_wins, old_wins, draws

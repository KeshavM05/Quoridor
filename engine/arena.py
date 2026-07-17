"""
Arena: pit two models against each other using C++ MCTS for speed.
"""

import numpy as np
import torch
from game import QuoridorGame
from model import QuoridorNet, action_to_move, ACTION_SIZE

try:
    import quoridor_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


def _make_eval_fn(model, device):
    def eval_batch(states_np):
        t = torch.FloatTensor(np.array(states_np)).to(device)
        model.eval()
        with torch.no_grad():
            p, v = model(t)
        return torch.softmax(p, dim=1).cpu().numpy(), v.squeeze(-1).cpu().numpy()
    return eval_batch


def play_one_game(model1, model2, device='cpu', num_simulations=50, max_moves=200):
    if not HAS_CPP:
        from mcts import MCTS
        game = QuoridorGame()
        mcts1 = MCTS(model1, device=device, num_simulations=num_simulations)
        mcts2 = MCTS(model2, device=device, num_simulations=num_simulations)
        move_count = 0
        while game.get_winner() == 0 and move_count < max_moves:
            if game.current_player == 1:
                pi = mcts1.search(game, temperature=0.1, add_noise=False)
            else:
                pi = mcts2.search(game, temperature=0.1, add_noise=False)
            action = np.argmax(pi)
            move = action_to_move(action)
            if move not in game.get_legal_moves():
                legal = game.get_legal_moves()
                if not legal:
                    break
                move = legal[0]
            game.play_move(move)
            move_count += 1
        return game.get_winner()

    # C++ fast path
    game = quoridor_cpp.QuoridorGame()
    eval_fn1 = _make_eval_fn(model1, device)
    eval_fn2 = _make_eval_fn(model2, device)
    move_count = 0

    while game.get_winner() == 0 and move_count < max_moves:
        if game.current_player == 1:
            pi = quoridor_cpp.mcts_search(game, num_simulations, 8, 0.1, False, eval_fn1)
        else:
            pi = quoridor_cpp.mcts_search(game, num_simulations, 8, 0.1, False, eval_fn2)
        action = int(np.argmax(pi))
        move = quoridor_cpp.Move.from_action(action)
        game.play_move(move)
        move_count += 1

    return game.get_winner()


def pit_models(new_model, old_model, device='cpu', num_games=40, num_simulations=50):
    new_wins = 0
    old_wins = 0
    draws = 0

    for i in range(num_games):
        if i % 2 == 0:
            result = play_one_game(new_model, old_model, device=device, num_simulations=num_simulations)
            if result == 1:
                new_wins += 1
            elif result == 2:
                old_wins += 1
            else:
                draws += 1
        else:
            result = play_one_game(old_model, new_model, device=device, num_simulations=num_simulations)
            if result == 2:
                new_wins += 1
            elif result == 1:
                old_wins += 1
            else:
                draws += 1

        if (i + 1) % 10 == 0:
            print(f'    Arena: {i+1}/{num_games} — new: {new_wins}, old: {old_wins}, draws: {draws}')

    return new_wins, old_wins, draws

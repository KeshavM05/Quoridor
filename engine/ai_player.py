"""
AI player that can be used by the FastAPI server to make moves.
Loads a trained model and uses MCTS to pick moves.
"""

import os
import numpy as np
import torch
from model import QuoridorNet, action_to_move, move_to_action, ACTION_SIZE
from mcts import MCTS


class AIPlayer:
    def __init__(self, checkpoint_path=None, device=None, num_simulations=200):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = QuoridorNet()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"AI: Loaded model from {checkpoint_path}")
        else:
            print("AI: Using untrained model (random play)")

        self.model.to(self.device)
        self.model.eval()
        self.num_simulations = num_simulations

    def get_move(self, game, temperature=0.1):
        """
        Given a game state, return the best move.
        """
        mcts = MCTS(self.model, device=self.device, num_simulations=self.num_simulations)
        pi = mcts.search(game, temperature=temperature, add_noise=False)

        # Pick highest probability legal action
        legal_moves = game.get_legal_moves()
        legal_actions = [move_to_action(m) for m in legal_moves]

        best_action = max(legal_actions, key=lambda a: pi[a])
        return action_to_move(best_action)


# Singleton for the server
_ai_instance = None

def get_ai(checkpoint_path=None, num_simulations=200):
    global _ai_instance
    if _ai_instance is None:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pt')
        _ai_instance = AIPlayer(checkpoint_path=checkpoint_path, num_simulations=num_simulations)
    return _ai_instance

"""
Monte Carlo Tree Search guided by a neural network (AlphaZero style).

Each node stores:
  - N(s,a): visit count
  - W(s,a): total value
  - Q(s,a): mean value = W/N
  - P(s,a): prior probability from the network

Selection uses PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
"""

import numpy as np
import copy
from model import encode_state, move_to_action, action_to_move, get_legal_action_mask, ACTION_SIZE


C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'children', 'visit_count', 'total_value', 'game_state', 'is_expanded']

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


class MCTS:
    def __init__(self, model, device='cpu', num_simulations=100):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    def search(self, game, temperature=1.0, add_noise=True):
        """
        Run MCTS from the given game state.
        Returns: action probabilities (pi) over the full action space.
        """
        import torch

        root = MCTSNode()
        root.game_state = copy.deepcopy(game)
        root.is_expanded = True

        # Get network prediction for root
        policy, value = self._evaluate(root.game_state)

        # Add Dirichlet noise to root for exploration
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

        # Create children for root
        legal_moves = root.game_state.get_legal_moves()
        for move in legal_moves:
            action = move_to_action(move)
            child = MCTSNode(parent=root, action=action, prior=policy[action])
            root.children[action] = child

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Select
            while node.is_expanded and node.children:
                node = self._select_child(node)
                search_path.append(node)

            # Check terminal
            if node.game_state is None:
                node.game_state = copy.deepcopy(search_path[-2].game_state)
                move = action_to_move(node.action)
                node.game_state.play_move(move)

            winner = node.game_state.get_winner()
            if winner != 0:
                # Terminal node
                # Value from perspective of the node's parent's player
                parent_player = search_path[-2].game_state.current_player
                value = 1.0 if winner == parent_player else -1.0
            else:
                # Expand
                policy, value = self._evaluate(node.game_state)
                legal_mask = get_legal_action_mask(node.game_state)
                policy = policy * legal_mask
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy /= policy_sum

                legal_moves = node.game_state.get_legal_moves()
                for move in legal_moves:
                    action = move_to_action(move)
                    child = MCTSNode(parent=node, action=action, prior=policy[action])
                    node.children[action] = child
                node.is_expanded = True

                # Value is from current player's perspective
                # We need to negate since backprop alternates
                value = -value

            # Backpropagate
            for node in reversed(search_path):
                node.visit_count += 1
                node.total_value += value
                value = -value  # Flip for alternating perspective

        # Compute action probabilities from visit counts
        action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        if temperature == 0:
            # Greedy
            best = np.argmax(action_probs)
            action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            action_probs[best] = 1.0
        else:
            # Apply temperature
            if action_probs.sum() > 0:
                action_probs = action_probs ** (1.0 / temperature)
                action_probs /= action_probs.sum()

        return action_probs

    def _select_child(self, node):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            score = child.ucb_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _evaluate(self, game):
        """Run the neural network on a game state. Returns (policy, value)."""
        import torch

        state = encode_state(game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()

        return policy, value

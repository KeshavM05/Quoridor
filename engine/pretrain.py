"""
Supervised pre-training: teach the model to follow BFS shortest path.

Instead of starting from random (where the model can't even walk straight),
we first train it on a simple heuristic: "always move toward goal via the
shortest path." This gives the model a baseline that:
- Knows moving forward is good
- Knows which direction to go
- Provides a foundation for self-play to build strategy on top of

Takes ~5 minutes to generate data and train. Then self-play starts from
a model that already plays basic Quoridor instead of random garbage.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
import os
import time

from game import QuoridorGame
from model import QuoridorNet, encode_state, move_to_action, action_to_move, get_legal_action_mask, ACTION_SIZE


def bfs_shortest_path_move(game):
    """
    Find the move that puts the current player on the shortest BFS path to goal.
    Returns the best pawn move (ignores walls for pre-training).
    """
    if game.current_player == 1:
        pos = game.p1_pos
        target_row = 8
    else:
        pos = game.p2_pos
        target_row = 0

    # BFS from current position to find shortest path
    parent = {}
    q = deque([pos])
    visited = set([pos])
    found = None

    while q:
        r, c = q.popleft()
        if r == target_row:
            found = (r, c)
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                if not game._is_blocked(r, c, nr, nc):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))

    if found is None:
        return None

    # Trace back to find first move
    curr = found
    while curr in parent and parent[curr] != pos:
        curr = parent[curr]

    if curr == pos or curr not in parent:
        return None

    # curr is the cell we should move to
    return ('move', curr[0], curr[1])


def generate_pretrain_data(num_games=5000, max_moves=50):
    """
    Generate supervised training data by playing games with BFS-optimal moves.
    Both players follow shortest path (no walls). This creates examples of
    efficient, goal-directed play.

    Also generates some games WITH random walls placed (by opponent) to teach
    the model to navigate around obstacles.
    """
    print(f"Generating {num_games} pre-training games...")
    examples = []

    for game_idx in range(num_games):
        game = QuoridorGame()
        move_count = 0

        # 50% of games: randomly place some walls first (to learn wall navigation)
        if game_idx % 2 == 0:
            num_random_walls = random.randint(1, 6)
            for _ in range(num_random_walls):
                wall_moves = [m for m in game.get_legal_moves() if m[0] == 'wall']
                if wall_moves:
                    wall = random.choice(wall_moves)
                    game.play_move(wall)

        while game.get_winner() == 0 and move_count < max_moves:
            # Get BFS optimal move
            best_move = bfs_shortest_path_move(game)

            if best_move is None or best_move not in game.get_legal_moves():
                # Fallback: pick any legal pawn move toward goal
                legal = game.get_legal_moves()
                pawn_moves = [m for m in legal if m[0] == 'move']
                if not pawn_moves:
                    break
                best_move = random.choice(pawn_moves)

            # Create training example
            state = encode_state(game)

            # Policy: put 80% weight on best move, 20% spread on other legal moves
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            legal_mask = get_legal_action_mask(game)
            best_action = move_to_action(best_move)
            policy[best_action] = 0.8

            # Spread remaining 0.2 across other legal pawn moves toward goal
            legal_pawn = [m for m in game.get_legal_moves() if m[0] == 'move']
            for m in legal_pawn:
                a = move_to_action(m)
                if a != best_action:
                    policy[a] = 0.2 / max(1, len(legal_pawn) - 1)

            # Normalize
            if policy.sum() > 0:
                policy /= policy.sum()

            # Value: simple heuristic based on BFS distance
            p1_dist = _bfs_distance(game, game.p1_pos, 8)
            p2_dist = _bfs_distance(game, game.p2_pos, 0)
            if game.current_player == 1:
                value = (p2_dist - p1_dist) / 16.0
            else:
                value = (p1_dist - p2_dist) / 16.0
            value = max(-1.0, min(1.0, value))

            examples.append((state, policy, value))

            game.play_move(best_move)
            move_count += 1

        if (game_idx + 1) % 500 == 0:
            print(f"  {game_idx + 1}/{num_games} games, {len(examples)} positions")

    print(f"  Generated {len(examples)} pre-training positions")
    return examples


def _bfs_distance(game, start_pos, target_row):
    """BFS shortest path distance."""
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
    return 50


class PretrainDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, value = self.examples[idx]
        return torch.FloatTensor(state), torch.FloatTensor(pi), torch.FloatTensor([value])


def pretrain(model, examples, device='cpu', epochs=20, batch_size=256, lr=0.002):
    """Train the model on supervised BFS data."""
    dataset = PretrainDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        batches = 0

        for states, pis, values in loader:
            states = states.to(device)
            pis = pis.to(device)
            values = values.to(device)

            policy_logits, pred_values = model(states)

            policy_loss = -torch.mean(torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1))
            value_loss = torch.mean((values - pred_values) ** 2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        avg_p = total_policy_loss / batches
        avg_v = total_value_loss / batches
        print(f"  Epoch {epoch+1}/{epochs} — policy_loss: {avg_p:.4f}, value_loss: {avg_v:.4f}")

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pre-train model on BFS shortest path')
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default='checkpoints/best_model.pt')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    t0 = time.time()
    examples = generate_pretrain_data(num_games=args.games, max_moves=50)
    print(f"Data generation: {time.time()-t0:.1f}s")

    # Train
    model = QuoridorNet()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    model = pretrain(model, examples, device=device, epochs=args.epochs)
    print(f"Training: {time.time()-t0:.1f}s")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved to {args.output}")

    # Quick test: does it play forward?
    model.eval()
    game = QuoridorGame()
    print("\nTest game (first 10 moves):")
    for i in range(10):
        state = encode_state(game)
        with torch.no_grad():
            policy_logits, value = model(torch.FloatTensor(state).unsqueeze(0).to(device))
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        legal_mask = get_legal_action_mask(game)
        policy = policy * legal_mask
        if policy.sum() > 0:
            policy /= policy.sum()
        action = np.argmax(policy)
        move = action_to_move(action)
        print(f"  P{game.current_player}: {move} (value: {value.item():.3f})")
        game.play_move(move)
        if game.get_winner() != 0:
            print(f"  Winner: Player {game.get_winner()}!")
            break

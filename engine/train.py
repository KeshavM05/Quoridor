"""
AlphaZero training loop.

Cycle:
  1. Self-play with current best model → generate training data
  2. Train neural network on accumulated data
  3. Pit new model vs old model in arena
  4. If new model wins >55%, it becomes the best model
  5. Repeat
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import time
import argparse

from model import QuoridorNet, ACTION_SIZE
from self_play import generate_self_play_data
from arena import pit_models


class QuoridorDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, pi, value = self.examples[idx]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(pi),
            torch.FloatTensor([value])
        )


def train_network(model, examples, device='cpu', epochs=10, batch_size=64, lr=0.001):
    """Train the network on self-play data."""
    dataset = QuoridorDataset(examples)
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

            # Policy loss: cross-entropy with MCTS policy
            policy_loss = -torch.mean(torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1))

            # Value loss: MSE
            value_loss = torch.mean((values - pred_values) ** 2)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        if batches > 0:
            print(f"  Epoch {epoch+1}/{epochs} — policy_loss: {total_policy_loss/batches:.4f}, value_loss: {total_value_loss/batches:.4f}")

    return model


def training_loop(
    num_iterations=50,
    num_self_play_games=100,
    num_simulations=100,
    num_arena_games=40,
    win_threshold=0.55,
    epochs=10,
    batch_size=64,
    lr=0.001,
    checkpoint_dir='checkpoints',
    device=None
):
    """Main AlphaZero training loop."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model
    model = QuoridorNet()
    model.to(device)

    # Check for existing checkpoint
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        print(f"Loading existing model from {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Training data buffer (keep last N games worth)
    replay_buffer = deque(maxlen=50000)

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*50}")

        # 1. Self-play
        print(f"\n[1/3] Self-play ({num_self_play_games} games, {num_simulations} sims/move)...")
        t0 = time.time()
        examples = generate_self_play_data(
            model, device=device,
            num_games=num_self_play_games,
            num_simulations=num_simulations
        )
        print(f"  Generated {len(examples)} training positions in {time.time()-t0:.1f}s")
        replay_buffer.extend(examples)

        # 2. Train
        print(f"\n[2/3] Training ({epochs} epochs, {len(replay_buffer)} positions)...")
        t0 = time.time()

        # Save current model for arena comparison
        old_state = {k: v.clone() for k, v in model.state_dict().items()}

        train_network(
            model, list(replay_buffer),
            device=device, epochs=epochs, batch_size=batch_size, lr=lr
        )
        print(f"  Training completed in {time.time()-t0:.1f}s")

        # 3. Arena
        print(f"\n[3/3] Arena ({num_arena_games} games)...")
        t0 = time.time()

        old_model = QuoridorNet()
        old_model.load_state_dict(old_state)
        old_model.to(device)

        new_wins, old_wins, draws = pit_models(
            model, old_model, device=device,
            num_games=num_arena_games,
            num_simulations=num_simulations // 2  # Fewer sims for speed
        )
        total_decisive = new_wins + old_wins
        win_rate = new_wins / total_decisive if total_decisive > 0 else 0.5
        print(f"  New model: {new_wins}W / {old_wins}L / {draws}D (win rate: {win_rate:.1%})")
        print(f"  Arena completed in {time.time()-t0:.1f}s")

        if win_rate >= win_threshold:
            print(f"  ✓ New model accepted (>{win_threshold:.0%})")
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'model_iter_{iteration}.pt'))
        else:
            print(f"  ✗ New model rejected — reverting")
            model.load_state_dict(old_state)

    print(f"\nTraining complete. Best model saved to {best_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AlphaZero for Quoridor')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--self-play-games', type=int, default=100)
    parser.add_argument('--simulations', type=int, default=100)
    parser.add_argument('--arena-games', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    training_loop(
        num_iterations=args.iterations,
        num_self_play_games=args.self_play_games,
        num_simulations=args.simulations,
        num_arena_games=args.arena_games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )

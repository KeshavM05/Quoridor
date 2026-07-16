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
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import argparse

from model import QuoridorNet, ACTION_SIZE
from self_play import generate_self_play_data
from arena import pit_models
from journal import TrainingJournal, select_notable_games


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


def train_network(model, examples, device='cpu', epochs=10, batch_size=64, lr=0.001, writer=None, global_step=0):
    """Train the network on self-play data.

    Returns:
        model: the trained model
        epoch_losses: list of (avg_policy_loss, avg_value_loss) per epoch
    """
    dataset = QuoridorDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    model.to(device)

    epoch_losses = []

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
            avg_policy = total_policy_loss / batches
            avg_value = total_value_loss / batches
            epoch_losses.append((avg_policy, avg_value))
            print(f"  Epoch {epoch+1}/{epochs} — policy_loss: {avg_policy:.4f}, value_loss: {avg_value:.4f}")

            if writer is not None:
                step = global_step * epochs + epoch
                writer.add_scalar('loss/policy', avg_policy, step)
                writer.add_scalar('loss/value', avg_value, step)

    return model, epoch_losses


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
    device=None,
    parallel=None,
    parallel_batch_size=16
):
    """Main AlphaZero training loop."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard logging
    writer = SummaryWriter(log_dir='runs')

    # Initialize training journal
    journal = TrainingJournal(config={
        'num_iterations': num_iterations,
        'num_self_play_games': num_self_play_games,
        'num_simulations': num_simulations,
        'num_arena_games': num_arena_games,
        'win_threshold': win_threshold,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'device': device,
    })

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

    training_start = time.time()

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*50}")

        # 1. Self-play (with replay recording for journal)
        print(f"\n[1/3] Self-play ({num_self_play_games} games, {num_simulations} sims/move)...")
        t0 = time.time()
        examples, avg_game_length, game_replays = generate_self_play_data(
            model, device=device,
            num_games=num_self_play_games,
            num_simulations=num_simulations,
            record_replays=True,
            parallel=parallel,
            batch_size=parallel_batch_size
        )
        print(f"  Generated {len(examples)} training positions in {time.time()-t0:.1f}s")
        replay_buffer.extend(examples)

        # Select notable games for the journal
        notable_games = select_notable_games(game_replays)

        # Log self-play metrics
        writer.add_scalar('self_play/avg_game_length', avg_game_length, iteration)
        writer.add_scalar('self_play/replay_buffer_size', len(replay_buffer), iteration)

        # 2. Train
        print(f"\n[2/3] Training ({epochs} epochs, {len(replay_buffer)} positions)...")
        t0 = time.time()

        # Save current model for arena comparison
        old_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Log learning rate
        writer.add_scalar('train/learning_rate', lr, iteration)

        model, epoch_losses = train_network(
            model, list(replay_buffer),
            device=device, epochs=epochs, batch_size=batch_size, lr=lr,
            writer=writer, global_step=iteration
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

        # Log arena win rate
        writer.add_scalar('arena/win_rate', win_rate, iteration)

        model_accepted = win_rate >= win_threshold
        if model_accepted:
            print(f"  ✓ New model accepted (>{win_threshold:.0%})")
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'model_iter_{iteration}.pt'))
        else:
            print(f"  ✗ New model rejected — reverting")
            model.load_state_dict(old_state)

        # Log iteration to journal
        policy_loss = float(epoch_losses[-1][0]) if epoch_losses else None
        value_loss = float(epoch_losses[-1][1]) if epoch_losses else None

        journal.log_iteration(
            iteration=iteration,
            policy_loss=policy_loss,
            value_loss=value_loss,
            win_rate=float(win_rate),
            avg_game_length=float(avg_game_length),
            model_accepted=model_accepted,
            notable_games=notable_games,
            num_games_played=num_self_play_games,
        )

        # Save journal checkpoint every 5 iterations
        if iteration % 5 == 0:
            journal.save_checkpoint(iteration, model.state_dict())

        # Save metrics for web dashboard
        from dashboard import save_metrics
        save_metrics({
            'iteration': iteration,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'win_rate': float(win_rate),
            'avg_game_length': float(avg_game_length),
            'total_positions': len(replay_buffer),
            'model_accepted': model_accepted,
        })

    # Finalize the journal with summary
    total_time = time.time() - training_start
    journal.finalize(total_time)

    writer.close()
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
    parser.add_argument('--parallel', action='store_true', default=None,
                        help='Force parallel self-play with batched inference (auto-detected on CUDA)')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                        help='Force sequential self-play (original behavior)')
    parser.add_argument('--parallel-batch-size', type=int, default=16,
                        help='Number of games to play simultaneously in parallel mode (default: 16)')
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
        checkpoint_dir=args.checkpoint_dir,
        parallel=args.parallel,
        parallel_batch_size=args.parallel_batch_size
    )

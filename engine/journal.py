"""
Training journal: auto-documenting system for AlphaZero training runs.

Records hyperparameters, per-iteration metrics, notable games, milestones,
and checkpoints. Generates a summary report at the end of training.

Usage:
    journal = TrainingJournal(config)
    # ... each iteration:
    journal.log_iteration(iteration, metrics, notable_games)
    # ... at end:
    journal.finalize(total_time)
"""

import os
import json
import time
import shutil
from datetime import datetime
from collections import deque


def _find_next_run_number(base_dir):
    """Find the next available run number in training_runs/."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not existing:
        return 1
    numbers = []
    for d in existing:
        try:
            numbers.append(int(d.split('_')[1]))
        except (IndexError, ValueError):
            pass
    return max(numbers) + 1 if numbers else 1


class TrainingJournal:
    """Records everything during an AlphaZero training run."""

    def __init__(self, config, base_dir=None):
        """
        Initialize a training journal for a new run.

        Args:
            config: dict of all hyperparameters
            base_dir: root directory for training_runs (defaults to engine/training_runs)
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), 'training_runs')

        run_num = _find_next_run_number(base_dir)
        self.run_dir = os.path.join(base_dir, f'run_{run_num:03d}')
        os.makedirs(self.run_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.replays_dir = os.path.join(self.run_dir, 'replays')
        os.makedirs(self.replays_dir, exist_ok=True)

        self.config = config
        self.start_time = time.time()
        self.start_datetime = datetime.now().isoformat()

        # Save config
        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'run_number': run_num,
                'started_at': self.start_datetime,
                **config
            }, f, indent=2)

        # Initialize journal file
        self.journal_path = os.path.join(self.run_dir, 'journal.jsonl')
        open(self.journal_path, 'w').close()  # Create empty file

        # Initialize milestones
        self.milestones_path = os.path.join(self.run_dir, 'milestones.json')
        self.milestones = []
        self._save_milestones()

        # Tracking state for milestone detection
        self._first_model_accepted = False
        self._first_short_win = False
        self._first_wall_on_shortest_path = False
        self._win_rate_70 = False
        self._win_rate_80 = False
        self._best_win_rate = 0.0
        self._iteration_metrics = []
        self._total_games = 0
        self._total_notable_games = []

        print(f"[Journal] Run directory: {self.run_dir}")

    def log_iteration(self, iteration, policy_loss, value_loss, win_rate,
                      avg_game_length, model_accepted, notable_games=None,
                      num_games_played=0):
        """
        Log one iteration's results to the journal.

        Args:
            iteration: iteration number
            policy_loss: final policy loss for this iteration
            value_loss: final value loss for this iteration
            win_rate: arena win rate (new model vs old)
            avg_game_length: average game length in self-play
            model_accepted: whether the new model was accepted
            notable_games: list of replay dicts (most decisive, longest, shortest)
            num_games_played: number of self-play games this iteration
        """
        timestamp = datetime.now().isoformat()
        self._total_games += num_games_played

        # Build journal entry
        entry = {
            'iteration': iteration,
            'timestamp': timestamp,
            'policy_loss': float(policy_loss) if policy_loss is not None else None,
            'value_loss': float(value_loss) if value_loss is not None else None,
            'win_rate': float(win_rate),
            'avg_game_length': float(avg_game_length),
            'model_accepted': model_accepted,
            'notable_games': [],
        }

        # Save notable games
        if notable_games:
            for i, game_replay in enumerate(notable_games):
                game_replay['iteration'] = iteration
                filename = f'iter_{iteration:03d}_game_{i}.json'
                filepath = os.path.join(self.replays_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(game_replay, f, indent=2)
                entry['notable_games'].append(filename)
                self._total_notable_games.append(game_replay)

        # Append to journal
        with open(self.journal_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Track metrics
        self._iteration_metrics.append(entry)
        if win_rate > self._best_win_rate:
            self._best_win_rate = win_rate

        # Detect milestones
        self._detect_milestones(iteration, win_rate, avg_game_length,
                                model_accepted, notable_games)

    def save_checkpoint(self, iteration, model_state_dict):
        """Save a model checkpoint (called every 5 iterations)."""
        import torch
        checkpoint_path = os.path.join(
            self.checkpoints_dir, f'checkpoint_iter_{iteration:03d}.pt'
        )
        torch.save(model_state_dict, checkpoint_path)
        print(f"[Journal] Checkpoint saved: {checkpoint_path}")

    def finalize(self, total_time=None):
        """
        Finalize the training run and generate summary.md.

        Args:
            total_time: total training time in seconds (if None, computed from start)
        """
        if total_time is None:
            total_time = time.time() - self.start_time

        summary = self._generate_summary(total_time)
        summary_path = os.path.join(self.run_dir, 'summary.md')
        with open(summary_path, 'w') as f:
            f.write(summary)

        print(f"[Journal] Training complete. Summary written to {summary_path}")
        return summary_path

    def _detect_milestones(self, iteration, win_rate, avg_game_length,
                           model_accepted, notable_games):
        """Check for and record training milestones."""
        # First model acceptance
        if model_accepted and not self._first_model_accepted:
            self._first_model_accepted = True
            self._add_milestone(iteration, 'first_model_accepted',
                                'First time the new model was accepted over the old model')

        # Win rate thresholds
        if win_rate >= 0.70 and not self._win_rate_70:
            self._win_rate_70 = True
            self._add_milestone(iteration, 'win_rate_70',
                                f'Win rate exceeded 70% ({win_rate:.1%})')

        if win_rate >= 0.80 and not self._win_rate_80:
            self._win_rate_80 = True
            self._add_milestone(iteration, 'win_rate_80',
                                f'Win rate exceeded 80% ({win_rate:.1%})')

        # Short game wins (under 20 moves)
        if notable_games and not self._first_short_win:
            for game in notable_games:
                if game.get('winner', 0) != 0 and game.get('length', 999) < 20:
                    self._first_short_win = True
                    self._add_milestone(
                        iteration, 'first_short_win',
                        f'First decisive win in under 20 moves '
                        f'(length: {game["length"]})')
                    break

        # Wall on shortest path detection
        if notable_games and not self._first_wall_on_shortest_path:
            for game in notable_games:
                if self._check_wall_on_shortest_path(game):
                    self._first_wall_on_shortest_path = True
                    self._add_milestone(
                        iteration, 'wall_on_shortest_path',
                        'First wall placement detected on opponent shortest path')
                    break

        # Average game length dropping significantly (games getting more efficient)
        if len(self._iteration_metrics) >= 5:
            early_avg = sum(m['avg_game_length'] for m in self._iteration_metrics[:3]) / 3
            recent_avg = sum(m['avg_game_length'] for m in self._iteration_metrics[-3:]) / 3
            if recent_avg < early_avg * 0.7 and not hasattr(self, '_efficiency_milestone'):
                self._efficiency_milestone = True
                self._add_milestone(
                    iteration, 'games_getting_shorter',
                    f'Average game length dropped 30%+ '
                    f'(from {early_avg:.1f} to {recent_avg:.1f} moves)')

    def _check_wall_on_shortest_path(self, game_replay):
        """
        Heuristic: check if any wall move in the game was placed on a cell
        that could block the opponent's direct forward path.
        """
        moves = game_replay.get('moves', [])
        for move_entry in moves:
            move_tuple = move_entry.get('move_tuple')
            if move_tuple and move_tuple[0] == 'wall':
                # A wall between rows 2-6 and columns 3-5 is likely on
                # the opponent's shortest path (center corridor)
                _, orient, r, c = move_tuple
                if 2 <= r <= 5 and 2 <= c <= 5:
                    return True
        return False

    def _add_milestone(self, iteration, milestone_type, description):
        """Add a milestone to the milestones log."""
        milestone = {
            'iteration': iteration,
            'type': milestone_type,
            'description': description,
            'timestamp': datetime.now().isoformat(),
        }
        self.milestones.append(milestone)
        self._save_milestones()
        print(f"[Journal] MILESTONE (iter {iteration}): {description}")

    def _save_milestones(self):
        """Write milestones to disk."""
        with open(self.milestones_path, 'w') as f:
            json.dump(self.milestones, f, indent=2)

    def _generate_summary(self, total_time):
        """Generate a markdown summary of the training run."""
        hours = total_time / 3600
        minutes = (total_time % 3600) / 60

        metrics = self._iteration_metrics
        if not metrics:
            return "# Training Summary\n\nNo iterations completed.\n"

        # Compute statistics
        final_policy_loss = metrics[-1]['policy_loss']
        final_value_loss = metrics[-1]['value_loss']
        best_win_rate = self._best_win_rate
        num_accepted = sum(1 for m in metrics if m['model_accepted'])
        total_iterations = len(metrics)

        # Game length trends
        early_lengths = [m['avg_game_length'] for m in metrics[:max(1, len(metrics)//4)]]
        late_lengths = [m['avg_game_length'] for m in metrics[-max(1, len(metrics)//4):]]
        early_avg_len = sum(early_lengths) / len(early_lengths) if early_lengths else 0
        late_avg_len = sum(late_lengths) / len(late_lengths) if late_lengths else 0

        # Policy loss trend
        early_policy = [m['policy_loss'] for m in metrics[:max(1, len(metrics)//4)]
                        if m['policy_loss'] is not None]
        late_policy = [m['policy_loss'] for m in metrics[-max(1, len(metrics)//4):]
                       if m['policy_loss'] is not None]
        early_avg_policy = sum(early_policy) / len(early_policy) if early_policy else 0
        late_avg_policy = sum(late_policy) / len(late_policy) if late_policy else 0

        # Wall usage analysis from notable games
        wall_counts_early = []
        wall_counts_late = []
        for game in self._total_notable_games:
            walls_in_game = sum(1 for m in game.get('moves', [])
                                if m.get('move_tuple', [None])[0] == 'wall')
            if game.get('iteration', 0) <= total_iterations // 2:
                wall_counts_early.append(walls_in_game)
            else:
                wall_counts_late.append(walls_in_game)

        avg_walls_early = (sum(wall_counts_early) / len(wall_counts_early)
                           if wall_counts_early else 0)
        avg_walls_late = (sum(wall_counts_late) / len(wall_counts_late)
                          if wall_counts_late else 0)

        # Build summary
        lines = []
        lines.append("# Training Run Summary")
        lines.append("")
        lines.append(f"**Run started:** {self.start_datetime}")
        lines.append(f"**Duration:** {int(hours)}h {int(minutes)}m")
        lines.append(f"**Total iterations:** {total_iterations}")
        lines.append(f"**Total self-play games:** {self._total_games}")
        lines.append("")

        lines.append("## Configuration")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, value in self.config.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## Final Strength")
        lines.append("")
        lines.append(f"- **Best win rate:** {best_win_rate:.1%}")
        lines.append(f"- **Final policy loss:** {final_policy_loss:.4f}"
                     if final_policy_loss else "- **Final policy loss:** N/A")
        lines.append(f"- **Final value loss:** {final_value_loss:.4f}"
                     if final_value_loss else "- **Final value loss:** N/A")
        lines.append(f"- **Models accepted:** {num_accepted}/{total_iterations} "
                     f"({num_accepted/total_iterations:.0%})")
        lines.append("")

        lines.append("## Training Progression")
        lines.append("")
        lines.append(f"- Policy loss: {early_avg_policy:.4f} (early) -> "
                     f"{late_avg_policy:.4f} (late)")
        lines.append(f"- Average game length: {early_avg_len:.1f} (early) -> "
                     f"{late_avg_len:.1f} (late)")
        if early_avg_len > 0:
            efficiency_change = (late_avg_len - early_avg_len) / early_avg_len * 100
            if efficiency_change < 0:
                lines.append(f"- Path efficiency improved by "
                             f"{abs(efficiency_change):.0f}% "
                             f"(games getting shorter)")
            else:
                lines.append(f"- Games got {efficiency_change:.0f}% longer "
                             f"(more complex play)")
        lines.append("")

        lines.append("## Strategy Observations")
        lines.append("")
        if avg_walls_early > 0 or avg_walls_late > 0:
            if avg_walls_late > avg_walls_early * 1.3:
                lines.append("- Wall usage increased over training "
                             f"({avg_walls_early:.1f} -> {avg_walls_late:.1f} per game), "
                             "suggesting the model learned defensive/blocking strategies")
            elif avg_walls_late < avg_walls_early * 0.7:
                lines.append("- Wall usage decreased over training "
                             f"({avg_walls_early:.1f} -> {avg_walls_late:.1f} per game), "
                             "suggesting the model favors rushing over blocking")
            else:
                lines.append(f"- Wall usage remained stable "
                             f"(~{(avg_walls_early + avg_walls_late)/2:.1f} per game)")
        else:
            lines.append("- Insufficient notable game data for strategy analysis")

        if late_avg_len < 30:
            lines.append("- Late-training games are short (<30 moves), "
                         "indicating aggressive/direct play style")
        elif late_avg_len > 60:
            lines.append("- Late-training games are long (>60 moves), "
                         "indicating defensive/wall-heavy play")

        lines.append("")

        lines.append("## Milestones")
        lines.append("")
        if self.milestones:
            for ms in self.milestones:
                lines.append(f"- **Iteration {ms['iteration']}:** {ms['description']}")
        else:
            lines.append("- No milestones reached")
        lines.append("")

        lines.append("## Notable Games")
        lines.append("")
        # Find the shortest winning game overall
        shortest_win = None
        longest_game = None
        for game in self._total_notable_games:
            if game.get('winner', 0) != 0:
                if shortest_win is None or game['length'] < shortest_win['length']:
                    shortest_win = game
            if longest_game is None or game.get('length', 0) > longest_game.get('length', 0):
                longest_game = game

        if shortest_win:
            lines.append(f"- **Shortest win:** {shortest_win['length']} moves "
                         f"(iteration {shortest_win.get('iteration', '?')}, "
                         f"winner: Player {shortest_win['winner']})")
        if longest_game:
            lines.append(f"- **Longest game:** {longest_game.get('length', '?')} moves "
                         f"(iteration {longest_game.get('iteration', '?')})")
        lines.append(f"- **Total notable games saved:** {len(self._total_notable_games)}")
        lines.append("")

        return '\n'.join(lines)


def select_notable_games(game_replays):
    """
    From a list of game replays, select the 2-3 most interesting ones:
    - The most decisive win (shortest winning game)
    - The longest game
    - The shortest game (if different from most decisive)

    Args:
        game_replays: list of replay dicts with keys: moves, winner, length

    Returns:
        list of up to 3 notable replay dicts
    """
    if not game_replays:
        return []

    notable = []

    # Most decisive win: shortest game with a winner
    wins = [g for g in game_replays if g.get('winner', 0) != 0]
    if wins:
        most_decisive = min(wins, key=lambda g: g['length'])
        notable.append(most_decisive)

    # Longest game
    longest = max(game_replays, key=lambda g: g.get('length', 0))
    if longest not in notable:
        notable.append(longest)

    # Shortest game overall (may be same as most decisive)
    shortest = min(game_replays, key=lambda g: g.get('length', 0))
    if shortest not in notable:
        notable.append(shortest)

    return notable[:3]

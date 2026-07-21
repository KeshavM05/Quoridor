"""
Strong heuristic bot for Quoridor — no neural network needed.

Based on gorisanson's approach (90% win rate against minimax):
- Guided rollout: 70% follow BFS shortest path, 30% place filtered wall
- Wall placement: only walls that increase opponent's BFS distance
- Uses the C++ engine for speed

This bot serves two purposes:
1. A strong opponent to play against (for the "vs Computer" mode)
2. A data generator for supervised pre-training of the neural network
"""

import numpy as np
import random
from collections import deque
from game import QuoridorGame
from model import encode_state, move_to_action, ACTION_SIZE

try:
    import quoridor_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


def _bfs_distance_py(game, start_pos, target_row):
    """BFS shortest path distance (Python game engine)."""
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


def _get_bfs_move_py(game):
    """Get the move that follows BFS shortest path (Python)."""
    if game.current_player == 1:
        pos = game.p1_pos
        target_row = 8
    else:
        pos = game.p2_pos
        target_row = 0

    # BFS to find shortest path
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

    # Trace back to find first step
    curr = found
    while curr in parent and parent[curr] != pos:
        curr = parent[curr]

    if curr == pos:
        return None

    return ('move', curr[0], curr[1])


def _get_best_wall_py(game):
    """Find the wall that maximizes opponent's BFS distance increase (Python)."""
    if game.current_player == 1:
        opp_pos = game.p2_pos
        opp_target = 0
    else:
        opp_pos = game.p1_pos
        opp_target = 8

    current_dist = _bfs_distance_py(game, opp_pos, opp_target)
    walls_left = game.p1_walls if game.current_player == 1 else game.p2_walls

    if walls_left == 0:
        return None

    best_wall = None
    best_increase = 0

    # Only check walls near opponent (Chebyshev distance <= 2)
    opp_r, opp_c = opp_pos
    for r in range(8):
        for c in range(8):
            if max(abs(r - opp_r), abs(c - opp_c)) > 3:
                continue

            # Try horizontal wall
            if not game.h_walls[r, c]:
                overlap = False
                if c > 0 and game.h_walls[r, c - 1]:
                    overlap = True
                if c < 7 and game.h_walls[r, c + 1]:
                    overlap = True
                if game.v_walls[r, c]:
                    overlap = True

                if not overlap:
                    game.h_walls[r, c] = True
                    if game._path_exists(game.p1_pos, 8) and game._path_exists(game.p2_pos, 0):
                        new_dist = _bfs_distance_py(game, opp_pos, opp_target)
                        increase = new_dist - current_dist
                        if increase > best_increase:
                            best_increase = increase
                            best_wall = ('wall', 'h', r, c)
                    game.h_walls[r, c] = False

            # Try vertical wall
            if not game.v_walls[r, c]:
                overlap = False
                if r > 0 and game.v_walls[r - 1, c]:
                    overlap = True
                if r < 7 and game.v_walls[r + 1, c]:
                    overlap = True
                if game.h_walls[r, c]:
                    overlap = True

                if not overlap:
                    game.v_walls[r, c] = True
                    if game._path_exists(game.p1_pos, 8) and game._path_exists(game.p2_pos, 0):
                        new_dist = _bfs_distance_py(game, opp_pos, opp_target)
                        increase = new_dist - current_dist
                        if increase > best_increase:
                            best_increase = increase
                            best_wall = ('wall', 'v', r, c)
                    game.v_walls[r, c] = False

    return best_wall if best_increase >= 2 else None


def heuristic_move_py(game):
    """
    Pick a move using the heuristic strategy:
    - 70% of the time: follow BFS shortest path
    - 30% of the time: place a wall that maximizes opponent's path increase
    - Smart wall timing: only wall if opponent is closer to goal than us
    """
    my_pos = game.p1_pos if game.current_player == 1 else game.p2_pos
    opp_pos = game.p2_pos if game.current_player == 1 else game.p1_pos
    my_target = 8 if game.current_player == 1 else 0
    opp_target = 0 if game.current_player == 1 else 8

    my_dist = _bfs_distance_py(game, my_pos, my_target)
    opp_dist = _bfs_distance_py(game, opp_pos, opp_target)
    walls_left = game.p1_walls if game.current_player == 1 else game.p2_walls

    # Strategic wall decision: wall if opponent is closer to goal OR 30% random
    should_wall = False
    if walls_left > 0:
        if opp_dist <= my_dist:
            should_wall = random.random() < 0.6  # 60% wall when behind
        else:
            should_wall = random.random() < 0.2  # 20% wall when ahead

    if should_wall:
        wall = _get_best_wall_py(game)
        if wall and wall in game.get_legal_moves():
            return wall

    # Default: follow shortest path
    move = _get_bfs_move_py(game)
    if move and move in game.get_legal_moves():
        return move

    # Fallback: any legal pawn move
    legal = game.get_legal_moves()
    pawn_moves = [m for m in legal if m[0] == 'move']
    if pawn_moves:
        return random.choice(pawn_moves)
    return legal[0] if legal else None


def generate_heuristic_games_cpp(num_games=50000, max_moves=80):
    """
    Generate training data using C++ engine (10-50x faster than Python).
    Heuristic bot plays against itself.
    """
    import time
    import math

    if not HAS_CPP:
        print("C++ backend not available, falling back to Python")
        return generate_heuristic_games_py(num_games, max_moves)

    print(f"Generating {num_games} heuristic bot games (C++ engine)...")
    t0 = time.time()
    all_examples = []
    game_lengths = []
    wins = {1: 0, 2: 0, 0: 0}

    for game_idx in range(num_games):
        game = quoridor_cpp.QuoridorGame()
        history = []  # (state, action, player)

        move_count = 0
        while game.get_winner() == 0 and move_count < max_moves:
            state = np.array(quoridor_cpp.encode_state(game))
            player = game.current_player

            # Heuristic decision: wall or move?
            move = _heuristic_move_cpp(game)
            if move is None:
                break

            action = move.to_action()
            history.append((state, action, player))
            game.play_move(move)
            move_count += 1

        winner = game.get_winner()
        wins[winner] += 1
        game_lengths.append(move_count)

        # Create training examples
        gamma = 0.98
        N = len(history)
        for t, (state, action, player) in enumerate(history):
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[action] = 1.0

            if winner == 0:
                p1_dist = game.bfs_distance(0, 8) if hasattr(game, 'bfs_distance') else 4
                p2_dist = game.bfs_distance(1, 0) if hasattr(game, 'bfs_distance') else 4
                v_cap = math.tanh((p2_dist - p1_dist) / 4.0)
                value = v_cap if player == 1 else -v_cap
            elif winner == player:
                value = 1.0 * (gamma ** (N - 1 - t))
            else:
                value = -1.0 * (gamma ** (N - 1 - t))

            all_examples.append((state, policy, value))

        if (game_idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            avg_len = sum(game_lengths[-1000:]) / 1000
            print(f"  {game_idx+1}/{num_games} games | avg: {avg_len:.1f} moves | "
                  f"P1={wins[1]} P2={wins[2]} D={wins[0]} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    avg_len = sum(game_lengths) / len(game_lengths)
    print(f"\nDone! {len(all_examples)} positions in {elapsed:.1f}s")
    print(f"Average game length: {avg_len:.1f} | P1={wins[1]} P2={wins[2]} D={wins[0]}")
    return all_examples


def _heuristic_move_cpp(game):
    """Pick a move using heuristic strategy with C++ game engine."""
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        return None

    pawn_moves = [m for m in legal_moves if m.type == quoridor_cpp.MoveType.PAWN]
    wall_moves = [m for m in legal_moves if m.type != quoridor_cpp.MoveType.PAWN]

    # Get positions
    p1_pos = game.p1_pos
    p2_pos = game.p2_pos
    current = game.current_player

    my_pos = p1_pos if current == 1 else p2_pos
    opp_pos = p2_pos if current == 1 else p1_pos
    my_target = 8 if current == 1 else 0
    opp_target = 0 if current == 1 else 8
    walls_left = game.p1_walls if current == 1 else game.p2_walls

    # Get BFS distances
    my_dist = abs(my_pos[0] - my_target)  # rough estimate
    opp_dist = abs(opp_pos[0] - opp_target)

    # Decide: wall or move?
    should_wall = False
    if walls_left > 0 and wall_moves:
        if opp_dist <= my_dist:
            should_wall = random.random() < 0.5
        else:
            should_wall = random.random() < 0.15

    if should_wall:
        # Pick wall near opponent that's in filtered set
        filtered = game.get_filtered_legal_actions()
        filtered_walls = [m for m in filtered if m.type != quoridor_cpp.MoveType.PAWN]
        if filtered_walls:
            # Pick the one closest to opponent
            opp_r, opp_c = opp_pos
            best_wall = min(filtered_walls,
                           key=lambda m: abs(m.row - opp_r) + abs(m.col - opp_c))
            return best_wall

    # Move: pick pawn move closest to goal
    if pawn_moves:
        target_row = my_target
        best = min(pawn_moves, key=lambda m: abs(m.row - target_row))
        return best

    return legal_moves[0]


def generate_heuristic_games(num_games=50000, max_moves=80):
    """Auto-select C++ or Python backend."""
    if HAS_CPP:
        return generate_heuristic_games_cpp(num_games, max_moves)
    return generate_heuristic_games_py(num_games, max_moves)


def generate_heuristic_games_py(num_games=50000, max_moves=80):
    """Python fallback (slow)."""
    import time
    import math

    print(f"Generating {num_games} heuristic bot games (Python - slow)...")
    t0 = time.time()
    all_examples = []
    game_lengths = []
    wins = {1: 0, 2: 0, 0: 0}

    for game_idx in range(num_games):
        game = QuoridorGame()
        history = []

        move_count = 0
        while game.get_winner() == 0 and move_count < max_moves:
            state = encode_state(game)
            move = heuristic_move_py(game)
            if move is None:
                break
            history.append((state, move, game.current_player))
            game.play_move(move)
            move_count += 1

        winner = game.get_winner()
        wins[winner] += 1
        game_lengths.append(move_count)

        gamma = 0.98
        N = len(history)
        for t, (state, move, player) in enumerate(history):
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            action = move_to_action(move)
            policy[action] = 1.0
            if winner == 0:
                p1_dist = _bfs_distance_py(game, game.p1_pos, 8)
                p2_dist = _bfs_distance_py(game, game.p2_pos, 0)
                v_cap = math.tanh((p2_dist - p1_dist) / 4.0)
                value = v_cap if player == 1 else -v_cap
            elif winner == player:
                value = 1.0 * (gamma ** (N - 1 - t))
            else:
                value = -1.0 * (gamma ** (N - 1 - t))
            all_examples.append((state, policy, value))

        if (game_idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            avg_len = sum(game_lengths[-1000:]) / 1000
            print(f"  {game_idx+1}/{num_games} | avg: {avg_len:.1f} | {elapsed:.1f}s")

    print(f"Done! {len(all_examples)} positions in {time.time()-t0:.1f}s")
    return all_examples


if __name__ == '__main__':
    import argparse
    import torch
    import os
    import time
    from model import QuoridorNet

    parser = argparse.ArgumentParser(description='Train from heuristic bot games')
    parser.add_argument('--games', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default='checkpoints/best_model.pt')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    examples = generate_heuristic_games(num_games=args.games, max_moves=80)

    # Train
    print(f"\nTraining neural network on {len(examples)} heuristic bot positions...")
    model = QuoridorNet()
    model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    from torch.utils.data import DataLoader, TensorDataset
    states = torch.FloatTensor(np.array([e[0] for e in examples]))
    policies = torch.FloatTensor(np.array([e[1] for e in examples]))
    values = torch.FloatTensor(np.array([[e[2]] for e in examples]))

    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()

    t0 = time.time()
    for epoch in range(args.epochs):
        total_p, total_v, n = 0, 0, 0
        for s, p, v in loader:
            s, p, v = s.to(device), p.to(device), v.to(device)
            pl, pv = model(s)
            ploss = -torch.mean(torch.sum(p * torch.log_softmax(pl, dim=1), dim=1))
            vloss = torch.mean((v - pv) ** 2)
            loss = ploss + vloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_p += ploss.item()
            total_v += vloss.item()
            n += 1
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: policy={total_p/n:.4f} value={total_v/n:.4f}")
    print(f"Training done in {time.time()-t0:.1f}s")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved to {args.output}")

    # Test: play 10 moves
    model.eval()
    game = QuoridorGame()
    print("\nTest game (model vs itself, first 20 moves):")
    for i in range(20):
        state = encode_state(game)
        with torch.no_grad():
            pl, pv = model(torch.FloatTensor(state).unsqueeze(0).to(device))
        policy = torch.softmax(pl, dim=1).squeeze(0).cpu().numpy()
        from model import get_legal_action_mask, action_to_move
        mask = get_legal_action_mask(game)
        policy *= mask
        if policy.sum() > 0:
            policy /= policy.sum()
        action = np.argmax(policy)
        move = action_to_move(action)
        print(f"  P{game.current_player}: {move} (val={pv.item():.3f})")
        game.play_move(move)
        if game.get_winner() != 0:
            print(f"  Winner: P{game.get_winner()}!")
            break

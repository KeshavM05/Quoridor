"""
Replay viewer: load, step through, and analyze recorded game replays.

Provides utilities for:
  - Stepping through a replay move by move, returning game states
  - Converting a replay to human-readable text
  - Generating heuristic-based summaries of what happened in a game
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from game import QuoridorGame


def load_replay(filepath):
    """Load a replay JSON file from disk.

    Args:
        filepath: path to a replay .json file

    Returns:
        replay dict with keys: moves, winner, length, iteration
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def replay_to_frames(replay):
    """
    Step through a replay move by move, returning a list of game states.

    Each frame is a dict with:
      - p1_pos: (row, col)
      - p2_pos: (row, col)
      - h_walls: list of (r, c) where horizontal walls are placed
      - v_walls: list of (r, c) where vertical walls are placed
      - current_player: whose turn it is next
      - move_number: which move produced this state (0 = initial)
      - last_move: the move that led to this state (None for initial)

    Args:
        replay: replay dict with 'moves' list

    Returns:
        list of frame dicts (length = len(moves) + 1 for initial state)
    """
    game = QuoridorGame()
    frames = []

    # Initial state (frame 0)
    frames.append(_game_to_frame(game, move_number=0, last_move=None))

    moves = replay.get('moves', [])
    for i, move_entry in enumerate(moves):
        move_tuple = move_entry.get('move_tuple')
        if move_tuple is None:
            continue

        # Convert list back to tuple format for the engine
        move_tuple = _normalize_move_tuple(move_tuple)
        game.play_move(move_tuple)

        frames.append(_game_to_frame(
            game,
            move_number=i + 1,
            last_move=move_entry
        ))

    return frames


def replay_to_text(replay):
    """
    Convert a replay to a formatted move-by-move text description.

    Args:
        replay: replay dict with 'moves' list

    Returns:
        Multi-line string with the full game notation
    """
    lines = []
    moves = replay.get('moves', [])
    winner = replay.get('winner', 0)
    length = replay.get('length', len(moves))
    iteration = replay.get('iteration', '?')

    lines.append(f"=== Game Replay (Iteration {iteration}) ===")
    lines.append(f"Length: {length} moves | Winner: {'Player ' + str(winner) if winner else 'Draw'}")
    lines.append("")

    # Format moves in pairs (like chess notation)
    move_num = 1
    i = 0
    while i < len(moves):
        p1_move = moves[i]
        line = f"{move_num:3d}. {_format_move_entry(p1_move)}"

        if i + 1 < len(moves):
            p2_move = moves[i + 1]
            line += f"    {_format_move_entry(p2_move)}"

        lines.append(line)
        move_num += 1
        i += 2

    lines.append("")
    if winner:
        player_name = "Red" if winner == 1 else "Blue"
        lines.append(f"Result: {player_name} (Player {winner}) wins")
    else:
        lines.append("Result: Draw (timeout)")

    return '\n'.join(lines)


def generate_replay_summary(replay):
    """
    Generate a heuristic-based natural language summary of a game.

    Analyzes:
      - Who won and how quickly
      - Wall placement patterns
      - Whether the winner took a direct path or detoured
      - Defensive vs aggressive play styles

    Args:
        replay: replay dict with 'moves' list

    Returns:
        String summary (1-3 sentences)
    """
    moves = replay.get('moves', [])
    winner = replay.get('winner', 0)
    length = replay.get('length', len(moves))

    if not moves:
        return "Empty game (no moves recorded)."

    # Analyze move composition per player
    p1_moves = [m for m in moves if m.get('player') == 1]
    p2_moves = [m for m in moves if m.get('player') == 2]

    p1_walls = sum(1 for m in p1_moves if _is_wall_move(m))
    p2_walls = sum(1 for m in p2_moves if _is_wall_move(m))
    p1_pawn = len(p1_moves) - p1_walls
    p2_pawn = len(p2_moves) - p2_walls

    # Player names
    winner_name = "Red" if winner == 1 else "Blue" if winner == 2 else None
    loser_name = "Blue" if winner == 1 else "Red" if winner == 2 else None

    parts = []

    # Describe the outcome
    if winner == 0:
        parts.append(f"The game ended in a draw after {length} moves (timeout)")
    elif length <= 12:
        parts.append(f"{winner_name} rushed straight to goal in just {length} moves")
    elif length <= 20:
        parts.append(f"{winner_name} won quickly in {length} moves")
    elif length <= 40:
        parts.append(f"{winner_name} won in {length} moves")
    else:
        parts.append(f"{winner_name} won after a long {length}-move battle")

    # Describe wall usage
    total_walls = p1_walls + p2_walls
    if total_walls == 0:
        parts.append("with no walls placed (pure pawn race)")
    elif total_walls <= 3:
        parts.append("with minimal wall usage")
    else:
        # Who used more walls?
        if winner == 1:
            winner_walls = p1_walls
            loser_walls = p2_walls
        elif winner == 2:
            winner_walls = p2_walls
            loser_walls = p1_walls
        else:
            winner_walls = 0
            loser_walls = 0

        if loser_walls > winner_walls + 2:
            parts.append(
                f"({loser_name} placed {loser_walls} walls but "
                f"{winner_name} found a path around)")
        elif winner_walls > loser_walls + 2:
            parts.append(
                f"({winner_name} used {winner_walls} walls to block "
                f"{loser_name}'s advance)")
        elif total_walls >= 6:
            parts.append(f"(heavy wall play: {p1_walls} by Red, {p2_walls} by Blue)")

    # Describe path efficiency
    if winner == 1 and p1_pawn > 0:
        # Player 1 needs to go from row 0 to row 8 = minimum 8 moves
        efficiency = 8 / p1_pawn if p1_pawn > 0 else 0
        if efficiency > 0.9:
            parts.append("- took an almost perfectly direct path")
        elif efficiency < 0.5:
            parts.append("- had to take a very indirect route")
    elif winner == 2 and p2_pawn > 0:
        # Player 2 needs to go from row 8 to row 0 = minimum 8 moves
        efficiency = 8 / p2_pawn if p2_pawn > 0 else 0
        if efficiency > 0.9:
            parts.append("- took an almost perfectly direct path")
        elif efficiency < 0.5:
            parts.append("- had to take a very indirect route")

    return ' '.join(parts) + '.'


def _game_to_frame(game, move_number, last_move):
    """Convert current game state to a serializable frame dict."""
    h_walls_list = []
    v_walls_list = []
    for r in range(8):
        for c in range(8):
            if game.h_walls[r, c]:
                h_walls_list.append([r, c])
            if game.v_walls[r, c]:
                v_walls_list.append([r, c])

    return {
        'p1_pos': list(game.p1_pos),
        'p2_pos': list(game.p2_pos),
        'p1_walls_remaining': game.p1_walls,
        'p2_walls_remaining': game.p2_walls,
        'h_walls': h_walls_list,
        'v_walls': v_walls_list,
        'current_player': game.current_player,
        'move_number': move_number,
        'last_move': last_move,
        'winner': game.get_winner(),
    }


def _normalize_move_tuple(move_tuple):
    """Convert a JSON-deserialized move (list) back to the tuple format the engine expects."""
    if isinstance(move_tuple, list):
        if move_tuple[0] == 'move':
            return ('move', move_tuple[1], move_tuple[2])
        elif move_tuple[0] == 'wall':
            return ('wall', move_tuple[1], move_tuple[2], move_tuple[3])
    return tuple(move_tuple) if isinstance(move_tuple, list) else move_tuple


def _format_move_entry(move_entry):
    """Format a single move entry for text display."""
    player = move_entry.get('player', '?')
    notation = move_entry.get('notation', '?')
    player_name = 'Red' if player == 1 else 'Blue'
    return f"{player_name}: {notation}"


def _is_wall_move(move_entry):
    """Check if a move entry is a wall placement."""
    move_tuple = move_entry.get('move_tuple')
    if move_tuple:
        if isinstance(move_tuple, list):
            return move_tuple[0] == 'wall'
        return move_tuple[0] == 'wall'
    # Fallback: check notation
    notation = move_entry.get('notation', '')
    return notation.startswith('h') or notation.startswith('v')


# CLI usage for viewing replays
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python replay.py <replay_file.json> [--frames]")
        sys.exit(1)

    filepath = sys.argv[1]
    replay = load_replay(filepath)

    if '--frames' in sys.argv:
        frames = replay_to_frames(replay)
        print(f"Total frames: {len(frames)}")
        for frame in frames:
            print(f"  Move {frame['move_number']}: "
                  f"P1={frame['p1_pos']} P2={frame['p2_pos']} "
                  f"Walls: {len(frame['h_walls'])}h {len(frame['v_walls'])}v")
    else:
        print(replay_to_text(replay))
        print()
        print("--- Summary ---")
        print(generate_replay_summary(replay))

"""
Test script to verify the C++ engine produces identical results to the Python engine.

Usage:
    cd engine
    python cpp/test_cpp.py

Requires: quoridor_cpp module built and importable (either via CMake or pip install).
"""

import sys
import os
import time
import numpy as np

# Add engine directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import QuoridorGame as PyGame
from model import encode_state as py_encode_state, move_to_action, action_to_move, get_legal_action_mask

try:
    import quoridor_cpp
except ImportError:
    print("ERROR: quoridor_cpp module not found.")
    print("Build it first:")
    print("  cd engine/cpp && pip install .")
    print("  OR")
    print("  cd engine/cpp && ./build.sh")
    sys.exit(1)


def test_initial_state():
    """Test that initial game state matches."""
    py_game = PyGame()
    cpp_game = quoridor_cpp.QuoridorGame()

    assert cpp_game.p1_pos == (0, 4), f"p1_pos: {cpp_game.p1_pos}"
    assert cpp_game.p2_pos == (8, 4), f"p2_pos: {cpp_game.p2_pos}"
    assert cpp_game.p1_walls == 10
    assert cpp_game.p2_walls == 10
    assert cpp_game.current_player == 1
    print("[PASS] Initial state matches")


def test_legal_pawn_moves():
    """Test that legal pawn moves match at the start."""
    py_game = PyGame()
    cpp_game = quoridor_cpp.QuoridorGame()

    py_moves = py_game._get_legal_pawn_moves()
    cpp_moves = cpp_game.get_legal_pawn_moves()

    py_actions = sorted([move_to_action(m) for m in py_moves])
    cpp_actions = sorted([m.to_action() for m in cpp_moves])

    assert py_actions == cpp_actions, f"Pawn moves differ:\n  Python: {py_actions}\n  C++:    {cpp_actions}"
    print(f"[PASS] Legal pawn moves match ({len(py_moves)} moves)")


def test_legal_wall_moves():
    """Test that legal wall moves match at the start."""
    py_game = PyGame()
    cpp_game = quoridor_cpp.QuoridorGame()

    py_moves = py_game._get_legal_wall_moves()
    cpp_moves = cpp_game.get_legal_wall_moves()

    py_actions = sorted([move_to_action(m) for m in py_moves])
    cpp_actions = sorted([m.to_action() for m in cpp_moves])

    assert py_actions == cpp_actions, f"Wall moves differ (Python: {len(py_moves)}, C++: {len(cpp_moves)})"
    print(f"[PASS] Legal wall moves match ({len(py_moves)} wall placements)")


def test_encode_state():
    """Test that state encoding matches."""
    py_game = PyGame()
    cpp_game = quoridor_cpp.QuoridorGame()

    py_state = py_encode_state(py_game)
    cpp_state = np.array(quoridor_cpp.encode_state(cpp_game))

    assert py_state.shape == cpp_state.shape == (12, 9, 9), \
        f"Shape mismatch: py={py_state.shape}, cpp={cpp_state.shape}"

    if not np.allclose(py_state, cpp_state, atol=1e-6):
        for plane in range(12):
            if not np.allclose(py_state[plane], cpp_state[plane], atol=1e-6):
                print(f"  Plane {plane} differs!")
                print(f"    Python:\n{py_state[plane]}")
                print(f"    C++:\n{cpp_state[plane]}")
        assert False, "encode_state mismatch"
    print("[PASS] encode_state matches exactly")


def test_play_moves():
    """Test that playing moves produces same state."""
    py_game = PyGame()
    cpp_game = quoridor_cpp.QuoridorGame()

    # Play a sequence of moves
    test_moves = [
        ('move', 1, 4),   # P1 moves south
        ('move', 7, 4),   # P2 moves north
        ('wall', 'h', 3, 3),  # P1 places horizontal wall
        ('move', 6, 4),   # P2 moves north
    ]

    for move in test_moves:
        py_game.play_move(move)

        if move[0] == 'move':
            cpp_move = quoridor_cpp.Move(quoridor_cpp.MoveType.PAWN, move[1], move[2])
        elif move[1] == 'h':
            cpp_move = quoridor_cpp.Move(quoridor_cpp.MoveType.WALL_H, move[2], move[3])
        else:
            cpp_move = quoridor_cpp.Move(quoridor_cpp.MoveType.WALL_V, move[2], move[3])
        cpp_game.play_move(cpp_move)

    assert cpp_game.p1_pos == py_game.p1_pos, f"p1_pos: cpp={cpp_game.p1_pos} py={py_game.p1_pos}"
    assert cpp_game.p2_pos == py_game.p2_pos, f"p2_pos: cpp={cpp_game.p2_pos} py={py_game.p2_pos}"
    assert cpp_game.p1_walls == py_game.p1_walls
    assert cpp_game.p2_walls == py_game.p2_walls
    assert cpp_game.current_player == py_game.current_player

    # Also verify legal moves match after these moves
    py_actions = sorted([move_to_action(m) for m in py_game.get_legal_moves()])
    cpp_actions = sorted([m.to_action() for m in cpp_game.get_legal_moves()])
    assert py_actions == cpp_actions, \
        f"Legal moves differ after playing sequence (Python: {len(py_actions)}, C++: {len(cpp_actions)})"

    print("[PASS] State matches after playing moves")


def test_wall_blocking():
    """Test wall blocking in specific scenarios."""
    cpp_game = quoridor_cpp.QuoridorGame()
    py_game = PyGame()

    # Place a horizontal wall at (3, 3) - blocks N/S movement
    cpp_game.set_h_wall(3, 3, True)
    py_game.h_walls[3, 3] = True

    # Check blocking in both implementations
    test_cases = [
        (3, 3, 4, 3, True),   # crossing the wall going S
        (3, 4, 4, 4, True),   # crossing the wall going S (second cell)
        (3, 2, 4, 2, False),  # not blocked (different col)
        (2, 3, 3, 3, False),  # not blocked (above the wall)
        (3, 3, 3, 4, False),  # E/W not blocked by H wall
    ]

    for r1, c1, r2, c2, expected in test_cases:
        cpp_result = cpp_game.is_blocked(r1, c1, r2, c2)
        py_result = py_game._is_blocked(r1, c1, r2, c2)
        assert cpp_result == py_result == expected, \
            f"is_blocked({r1},{c1},{r2},{c2}): cpp={cpp_result} py={py_result} expected={expected}"

    print("[PASS] Wall blocking logic matches")


def test_jump_logic():
    """Test jump scenarios (straight and diagonal)."""
    # Set up face-to-face scenario
    py_game = PyGame()
    py_game.p1_pos = (4, 4)
    py_game.p2_pos = (5, 4)
    py_game.current_player = 1

    cpp_game = quoridor_cpp.QuoridorGame()
    cpp_game.p1_pos = (4, 4)
    cpp_game.p2_pos = (5, 4)
    cpp_game.current_player = 1

    py_pawn = py_game._get_legal_pawn_moves()
    cpp_pawn = cpp_game.get_legal_pawn_moves()

    py_actions = sorted([move_to_action(m) for m in py_pawn])
    cpp_actions = sorted([m.to_action() for m in cpp_pawn])
    assert py_actions == cpp_actions, f"Jump (straight) differs:\n  py={py_actions}\n  cpp={cpp_actions}"

    # Now block the straight jump with a wall
    py_game.h_walls[5, 3] = True
    py_game.h_walls[5, 4] = True  # block both cells
    cpp_game.set_h_wall(5, 3, True)
    cpp_game.set_h_wall(5, 4, True)

    py_pawn = py_game._get_legal_pawn_moves()
    cpp_pawn = cpp_game.get_legal_pawn_moves()

    py_actions = sorted([move_to_action(m) for m in py_pawn])
    cpp_actions = sorted([m.to_action() for m in cpp_pawn])
    assert py_actions == cpp_actions, f"Jump (diagonal) differs:\n  py={py_actions}\n  cpp={cpp_actions}"

    print("[PASS] Jump logic (straight + diagonal) matches")


def test_path_exists():
    """Test BFS path finding."""
    cpp_game = quoridor_cpp.QuoridorGame()

    # No walls - path should exist
    assert cpp_game.path_exists((0, 4), 8) == True
    assert cpp_game.path_exists((8, 4), 0) == True

    # Place walls but don't fully block
    cpp_game.set_h_wall(0, 0, True)
    cpp_game.set_h_wall(0, 1, True)
    cpp_game.set_h_wall(0, 2, True)
    assert cpp_game.path_exists((0, 4), 8) == True  # can go around

    print("[PASS] Path existence (BFS) works correctly")


def benchmark():
    """Benchmark C++ vs Python for get_legal_moves."""
    print("\n--- Performance Benchmark ---")

    py_game = PyGame()
    cpp_game = quoridor_cpp.QuoridorGame()

    # Benchmark get_legal_moves
    N = 100

    start = time.perf_counter()
    for _ in range(N):
        py_game.get_legal_moves()
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(N):
        cpp_game.get_legal_moves()
    cpp_time = time.perf_counter() - start

    speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
    print(f"get_legal_moves ({N} calls):")
    print(f"  Python: {py_time*1000:.1f} ms")
    print(f"  C++:    {cpp_time*1000:.1f} ms")
    print(f"  Speedup: {speedup:.0f}x")

    # Benchmark encode_state
    start = time.perf_counter()
    for _ in range(N):
        py_encode_state(py_game)
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(N):
        quoridor_cpp.encode_state(cpp_game)
    cpp_time = time.perf_counter() - start

    speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
    print(f"\nencode_state ({N} calls):")
    print(f"  Python: {py_time*1000:.1f} ms")
    print(f"  C++:    {cpp_time*1000:.1f} ms")
    print(f"  Speedup: {speedup:.0f}x")

    # Benchmark game copy
    import copy
    start = time.perf_counter()
    for _ in range(1000):
        copy.deepcopy(py_game)
    py_time = time.perf_counter() - start

    import copy as copy_mod
    start = time.perf_counter()
    for _ in range(1000):
        copy_mod.copy(cpp_game)  # C++ copy via __copy__
    cpp_time = time.perf_counter() - start

    speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
    print(f"\ngame copy (1000 calls):")
    print(f"  Python deepcopy: {py_time*1000:.1f} ms")
    print(f"  C++ copy:        {cpp_time*1000:.1f} ms")
    print(f"  Speedup: {speedup:.0f}x")


if __name__ == "__main__":
    print("=== Testing C++ Quoridor Engine ===\n")

    test_initial_state()
    test_legal_pawn_moves()
    test_legal_wall_moves()
    test_encode_state()
    test_play_moves()
    test_wall_blocking()
    test_jump_logic()
    test_path_exists()

    print("\n=== All tests passed! ===")

    benchmark()

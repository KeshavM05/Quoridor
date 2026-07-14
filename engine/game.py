import numpy as np
from collections import deque
import copy

class QuoridorGame:
    def __init__(self):
        self.board_size = 9
        # Player 1 starts at (0, 4) and wants to reach row 8
        # Player 2 starts at (8, 4) and wants to reach row 0
        self.p1_pos = (0, 4)
        self.p2_pos = (8, 4)
        
        self.p1_walls = 10
        self.p2_walls = 10
        
        self.current_player = 1 # 1 or 2
        
        # Wall intersections: 8x8 grid.
        # h_walls[r][c] == 1 means horizontal wall between rows r, r+1 and cols c, c+1
        self.h_walls = np.zeros((8, 8), dtype=bool)
        self.v_walls = np.zeros((8, 8), dtype=bool)

    def get_legal_moves(self):
        """Returns a list of valid moves. A move is a tuple: ('move', r, c) or ('wall', 'h'|'v', r, c)"""
        moves = []
        moves.extend(self._get_legal_pawn_moves())
        moves.extend(self._get_legal_wall_moves())
        return moves
        
    def _get_legal_pawn_moves(self):
        """Calculates valid squares the current player can move to, handling jumps."""
        moves = []
        pos = self.p1_pos if self.current_player == 1 else self.p2_pos
        opp_pos = self.p2_pos if self.current_player == 1 else self.p1_pos
        
        r, c = pos
        # Directions: (dr, dc, wall_check_func)
        dirs = [
            (-1, 0), # N
            (1, 0),  # S
            (0, -1), # W
            (0, 1)   # E
        ]
        
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                # Check if wall blocks movement to nr, nc
                if not self._is_blocked(r, c, nr, nc):
                    if (nr, nc) == opp_pos:
                        # Jump logic
                        nnr, nnc = nr + dr, nc + dc
                        if 0 <= nnr < 9 and 0 <= nnc < 9 and not self._is_blocked(nr, nc, nnr, nnc):
                            # Straight jump
                            moves.append(('move', nnr, nnc))
                        else:
                            # Diagonal jump (if straight is blocked by edge or wall)
                            # We can jump to the sides of the opponent
                            if dr != 0: # Moving N/S, side jumps are E/W
                                for sdc in [-1, 1]:
                                    snr, snc = nr, nc + sdc
                                    if 0 <= snr < 9 and 0 <= snc < 9 and not self._is_blocked(nr, nc, snr, snc):
                                        moves.append(('move', snr, snc))
                            else: # Moving E/W, side jumps are N/S
                                for sdr in [-1, 1]:
                                    snr, snc = nr + sdr, nc
                                    if 0 <= snr < 9 and 0 <= snc < 9 and not self._is_blocked(nr, nc, snr, snc):
                                        moves.append(('move', snr, snc))
                    else:
                        moves.append(('move', nr, nc))
                        
        return moves

    def _is_blocked(self, r1, c1, r2, c2):
        """Checks if moving from (r1,c1) to (r2,c2) is blocked by a wall. Assumes adjacent."""
        if r1 == r2: # Moving E/W
            min_c = min(c1, c2)
            # Vertical wall blocks E/W movement
            # A v_wall at (wr, wc) blocks (wr, wc) to (wr, wc+1) AND (wr+1, wc) to (wr+1, wc+1)
            # So if we are moving from c to c+1 at row r, a vertical wall at wc=c and wr=r or r-1 blocks us.
            if min_c < 8:
                if (r1 < 8 and self.v_walls[r1, min_c]) or (r1 > 0 and self.v_walls[r1-1, min_c]):
                    return True
        elif c1 == c2: # Moving N/S
            min_r = min(r1, r2)
            # Horizontal wall blocks N/S movement
            if min_r < 8:
                if (c1 < 8 and self.h_walls[min_r, c1]) or (c1 > 0 and self.h_walls[min_r, c1-1]):
                    return True
        return False

    def _get_legal_wall_moves(self):
        walls_left = self.p1_walls if self.current_player == 1 else self.p2_walls
        if walls_left == 0:
            return []
            
        moves = []
        for r in range(8):
            for c in range(8):
                # Check H wall
                if not self.h_walls[r, c]:
                    # Check overlaps
                    overlap = False
                    if c > 0 and self.h_walls[r, c-1]: overlap = True
                    if c < 7 and self.h_walls[r, c+1]: overlap = True
                    if self.v_walls[r, c]: overlap = True # Crossing
                    
                    if not overlap:
                        # Check pathfinding (this is expensive, only do it if valid otherwise)
                        self.h_walls[r, c] = True
                        if self._path_exists(self.p1_pos, 8) and self._path_exists(self.p2_pos, 0):
                            moves.append(('wall', 'h', r, c))
                        self.h_walls[r, c] = False
                
                # Check V wall
                if not self.v_walls[r, c]:
                    overlap = False
                    if r > 0 and self.v_walls[r-1, c]: overlap = True
                    if r < 7 and self.v_walls[r+1, c]: overlap = True
                    if self.h_walls[r, c]: overlap = True
                    
                    if not overlap:
                        self.v_walls[r, c] = True
                        if self._path_exists(self.p1_pos, 8) and self._path_exists(self.p2_pos, 0):
                            moves.append(('wall', 'v', r, c))
                        self.v_walls[r, c] = False
        return moves

    def _path_exists(self, start_pos, target_row):
        """BFS to check if a player can reach their target row."""
        q = deque([start_pos])
        visited = set([start_pos])
        
        while q:
            r, c = q.popleft()
            if r == target_row:
                return True
                
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9 and (nr, nc) not in visited:
                    if not self._is_blocked(r, c, nr, nc):
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False

    def play_move(self, move):
        """Executes a move. Move format: ('move', r, c) or ('wall', 'h'|'v', r, c)"""
        if move[0] == 'move':
            if self.current_player == 1:
                self.p1_pos = (move[1], move[2])
            else:
                self.p2_pos = (move[1], move[2])
        elif move[0] == 'wall':
            _, orient, r, c = move
            if orient == 'h':
                self.h_walls[r, c] = True
            else:
                self.v_walls[r, c] = True
                
            if self.current_player == 1:
                self.p1_walls -= 1
            else:
                self.p2_walls -= 1
                
        self.current_player = 3 - self.current_player # Swap turn

    def get_winner(self):
        if self.p1_pos[0] == 8:
            return 1
        if self.p2_pos[0] == 0:
            return 2
        return 0

    def print_board(self):
        # A simple visualization for debugging
        for r in range(9):
            # Print cells and vertical walls
            row_str = ""
            for c in range(9):
                if self.p1_pos == (r, c):
                    row_str += "1"
                elif self.p2_pos == (r, c):
                    row_str += "2"
                else:
                    row_str += "."
                
                if c < 8:
                    if self._is_blocked(r, c, r, c+1):
                        row_str += "|"
                    else:
                        row_str += " "
            print(row_str)
            
            # Print horizontal walls
            if r < 8:
                wall_str = ""
                for c in range(9):
                    if self._is_blocked(r, c, r+1, c):
                        wall_str += "-"
                    else:
                        wall_str += " "
                    if c < 8:
                        wall_str += " "
                print(wall_str)

if __name__ == "__main__":
    game = QuoridorGame()
    print("Initial state:")
    game.print_board()
    print(f"P1 valid moves: {len(game.get_legal_moves())}")

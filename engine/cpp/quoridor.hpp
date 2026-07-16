#pragma once
/**
 * quoridor.hpp - Header-only C++ Quoridor game engine.
 *
 * Matches the Python QuoridorGame class exactly:
 *   - 9x9 board, 2 players
 *   - Player 1 starts at (0,4), goal row 8
 *   - Player 2 starts at (8,4), goal row 0
 *   - 10 walls each, h_walls[8][8], v_walls[8][8]
 *   - Wall blocking, BFS path validation, jump logic
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <queue>
#include <vector>

namespace quoridor {

static constexpr int BOARD_SIZE = 9;
static constexpr int WALL_GRID = 8;
static constexpr int ACTION_SIZE = 81 + 64 + 64; // 209

// Move types
enum class MoveType : uint8_t { PAWN = 0, WALL_H = 1, WALL_V = 2 };

struct Move {
    MoveType type;
    int8_t row;
    int8_t col;

    Move() : type(MoveType::PAWN), row(0), col(0) {}
    Move(MoveType t, int r, int c) : type(t), row(static_cast<int8_t>(r)), col(static_cast<int8_t>(c)) {}

    bool operator==(const Move& o) const {
        return type == o.type && row == o.row && col == o.col;
    }

    // Convert move to action index (0-208)
    int to_action() const {
        switch (type) {
            case MoveType::PAWN:
                return row * BOARD_SIZE + col;
            case MoveType::WALL_H:
                return 81 + row * WALL_GRID + col;
            case MoveType::WALL_V:
                return 81 + 64 + row * WALL_GRID + col;
        }
        return -1;
    }

    // Create move from action index
    static Move from_action(int action) {
        if (action < 81) {
            return Move(MoveType::PAWN, action / BOARD_SIZE, action % BOARD_SIZE);
        } else if (action < 81 + 64) {
            int idx = action - 81;
            return Move(MoveType::WALL_H, idx / WALL_GRID, idx % WALL_GRID);
        } else {
            int idx = action - 81 - 64;
            return Move(MoveType::WALL_V, idx / WALL_GRID, idx % WALL_GRID);
        }
    }
};

struct Position {
    int8_t row;
    int8_t col;

    Position() : row(0), col(0) {}
    Position(int r, int c) : row(static_cast<int8_t>(r)), col(static_cast<int8_t>(c)) {}

    bool operator==(const Position& o) const { return row == o.row && col == o.col; }
    bool operator!=(const Position& o) const { return !(*this == o); }
};

class QuoridorGame {
public:
    Position p1_pos;
    Position p2_pos;
    int8_t p1_walls;
    int8_t p2_walls;
    int8_t current_player; // 1 or 2

    // Wall grids: h_walls[r][c] = true means horizontal wall at intersection (r,c)
    bool h_walls[WALL_GRID][WALL_GRID];
    bool v_walls[WALL_GRID][WALL_GRID];

    QuoridorGame() {
        reset();
    }

    void reset() {
        p1_pos = Position(0, 4);
        p2_pos = Position(8, 4);
        p1_walls = 10;
        p2_walls = 10;
        current_player = 1;
        std::memset(h_walls, 0, sizeof(h_walls));
        std::memset(v_walls, 0, sizeof(v_walls));
    }

    // Copy constructor (fast memcpy)
    QuoridorGame(const QuoridorGame& other) = default;
    QuoridorGame& operator=(const QuoridorGame& other) = default;

    /**
     * Check if movement from (r1,c1) to adjacent (r2,c2) is blocked by a wall.
     * Matches Python's _is_blocked exactly.
     */
    bool is_blocked(int r1, int c1, int r2, int c2) const {
        if (r1 == r2) {
            // Moving East/West - vertical walls block
            int min_c = (c1 < c2) ? c1 : c2;
            if (min_c < 8) {
                if (r1 < 8 && v_walls[r1][min_c]) return true;
                if (r1 > 0 && v_walls[r1 - 1][min_c]) return true;
            }
        } else if (c1 == c2) {
            // Moving North/South - horizontal walls block
            int min_r = (r1 < r2) ? r1 : r2;
            if (min_r < 8) {
                if (c1 < 8 && h_walls[min_r][c1]) return true;
                if (c1 > 0 && h_walls[min_r][c1 - 1]) return true;
            }
        }
        return false;
    }

    /**
     * BFS to check if a position can reach the target row.
     * Used for wall validity checking.
     */
    bool path_exists(Position start, int target_row) const {
        // Visited array on stack (fast, no allocation)
        bool visited[BOARD_SIZE][BOARD_SIZE];
        std::memset(visited, 0, sizeof(visited));

        // Use a simple queue with fixed max size (81 cells)
        Position queue_buf[81];
        int front = 0, back = 0;

        queue_buf[back++] = start;
        visited[start.row][start.col] = true;

        static constexpr int dr[] = {-1, 1, 0, 0};
        static constexpr int dc[] = {0, 0, -1, 1};

        while (front < back) {
            Position pos = queue_buf[front++];
            if (pos.row == target_row) return true;

            for (int d = 0; d < 4; d++) {
                int nr = pos.row + dr[d];
                int nc = pos.col + dc[d];
                if (nr >= 0 && nr < 9 && nc >= 0 && nc < 9 && !visited[nr][nc]) {
                    if (!is_blocked(pos.row, pos.col, nr, nc)) {
                        visited[nr][nc] = true;
                        queue_buf[back++] = Position(nr, nc);
                    }
                }
            }
        }
        return false;
    }

    /**
     * Get all legal pawn moves for the current player, including jumps.
     */
    std::vector<Move> get_legal_pawn_moves() const {
        std::vector<Move> moves;
        moves.reserve(5); // max 4 normal + jumps

        Position pos = (current_player == 1) ? p1_pos : p2_pos;
        Position opp = (current_player == 1) ? p2_pos : p1_pos;

        static constexpr int dr[] = {-1, 1, 0, 0};
        static constexpr int dc[] = {0, 0, -1, 1};

        for (int d = 0; d < 4; d++) {
            int nr = pos.row + dr[d];
            int nc = pos.col + dc[d];

            if (nr < 0 || nr >= 9 || nc < 0 || nc >= 9) continue;
            if (is_blocked(pos.row, pos.col, nr, nc)) continue;

            if (nr == opp.row && nc == opp.col) {
                // Adjacent to opponent - jump logic
                int nnr = nr + dr[d];
                int nnc = nc + dc[d];

                if (nnr >= 0 && nnr < 9 && nnc >= 0 && nnc < 9 &&
                    !is_blocked(nr, nc, nnr, nnc)) {
                    // Straight jump
                    moves.emplace_back(MoveType::PAWN, nnr, nnc);
                } else {
                    // Diagonal jump (straight blocked by wall or edge)
                    if (dr[d] != 0) {
                        // Moving N/S, side jumps are E/W
                        for (int sdc : {-1, 1}) {
                            int snr = nr;
                            int snc = nc + sdc;
                            if (snr >= 0 && snr < 9 && snc >= 0 && snc < 9 &&
                                !is_blocked(nr, nc, snr, snc)) {
                                moves.emplace_back(MoveType::PAWN, snr, snc);
                            }
                        }
                    } else {
                        // Moving E/W, side jumps are N/S
                        for (int sdr : {-1, 1}) {
                            int snr = nr + sdr;
                            int snc = nc;
                            if (snr >= 0 && snr < 9 && snc >= 0 && snc < 9 &&
                                !is_blocked(nr, nc, snr, snc)) {
                                moves.emplace_back(MoveType::PAWN, snr, snc);
                            }
                        }
                    }
                }
            } else {
                // Normal move (no opponent in the way)
                moves.emplace_back(MoveType::PAWN, nr, nc);
            }
        }
        return moves;
    }

    /**
     * Get all legal wall placements for the current player.
     * Checks overlaps, crossings, and path existence (BFS).
     */
    std::vector<Move> get_legal_wall_moves() const {
        std::vector<Move> moves;

        int walls_left = (current_player == 1) ? p1_walls : p2_walls;
        if (walls_left == 0) return moves;

        moves.reserve(64); // rough estimate

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                // Check horizontal wall
                if (!h_walls[r][c]) {
                    bool overlap = false;
                    if (c > 0 && h_walls[r][c - 1]) overlap = true;
                    if (c < 7 && h_walls[r][c + 1]) overlap = true;
                    if (v_walls[r][c]) overlap = true; // crossing

                    if (!overlap) {
                        // Temporarily place wall and check paths
                        // const_cast is safe here since we restore immediately
                        auto* self = const_cast<QuoridorGame*>(this);
                        self->h_walls[r][c] = true;
                        if (path_exists(p1_pos, 8) && path_exists(p2_pos, 0)) {
                            moves.emplace_back(MoveType::WALL_H, r, c);
                        }
                        self->h_walls[r][c] = false;
                    }
                }

                // Check vertical wall
                if (!v_walls[r][c]) {
                    bool overlap = false;
                    if (r > 0 && v_walls[r - 1][c]) overlap = true;
                    if (r < 7 && v_walls[r + 1][c]) overlap = true;
                    if (h_walls[r][c]) overlap = true; // crossing

                    if (!overlap) {
                        auto* self = const_cast<QuoridorGame*>(this);
                        self->v_walls[r][c] = true;
                        if (path_exists(p1_pos, 8) && path_exists(p2_pos, 0)) {
                            moves.emplace_back(MoveType::WALL_V, r, c);
                        }
                        self->v_walls[r][c] = false;
                    }
                }
            }
        }
        return moves;
    }

    /**
     * Get all legal moves (pawn moves + wall placements).
     */
    std::vector<Move> get_legal_moves() const {
        std::vector<Move> pawn_moves = get_legal_pawn_moves();
        std::vector<Move> wall_moves = get_legal_wall_moves();

        std::vector<Move> all_moves;
        all_moves.reserve(pawn_moves.size() + wall_moves.size());
        all_moves.insert(all_moves.end(), pawn_moves.begin(), pawn_moves.end());
        all_moves.insert(all_moves.end(), wall_moves.begin(), wall_moves.end());
        return all_moves;
    }

    /**
     * Apply a move to the game state.
     */
    void play_move(const Move& move) {
        switch (move.type) {
            case MoveType::PAWN:
                if (current_player == 1) {
                    p1_pos = Position(move.row, move.col);
                } else {
                    p2_pos = Position(move.row, move.col);
                }
                break;
            case MoveType::WALL_H:
                h_walls[move.row][move.col] = true;
                if (current_player == 1) p1_walls--;
                else p2_walls--;
                break;
            case MoveType::WALL_V:
                v_walls[move.row][move.col] = true;
                if (current_player == 1) p1_walls--;
                else p2_walls--;
                break;
        }
        current_player = 3 - current_player; // Swap: 1->2, 2->1
    }

    /**
     * Check if there's a winner.
     * Returns 0 (no winner), 1 (player 1 wins), or 2 (player 2 wins).
     */
    int get_winner() const {
        if (p1_pos.row == 8) return 1;
        if (p2_pos.row == 0) return 2;
        return 0;
    }

    /**
     * Get the legal action mask (binary array of size 209).
     */
    std::array<float, ACTION_SIZE> get_legal_action_mask() const {
        std::array<float, ACTION_SIZE> mask{};
        std::vector<Move> moves = get_legal_moves();
        for (const auto& m : moves) {
            int action = m.to_action();
            if (action >= 0 && action < ACTION_SIZE) {
                mask[action] = 1.0f;
            }
        }
        return mask;
    }

    /**
     * Encode the game state into a 12x9x9 float array.
     * Matches Python's encode_state() exactly.
     * The state is from the perspective of the current player.
     */
    void encode_state(float* output) const {
        // Zero out all 12*9*9 = 972 floats
        std::memset(output, 0, 12 * BOARD_SIZE * BOARD_SIZE * sizeof(float));

        Position my_pos, opp_pos;
        int my_walls, opp_walls;
        int my_goal_row, opp_goal_row;

        if (current_player == 1) {
            my_pos = p1_pos;
            opp_pos = p2_pos;
            my_walls = p1_walls;
            opp_walls = p2_walls;
            my_goal_row = 8;
            opp_goal_row = 0;
        } else {
            my_pos = p2_pos;
            opp_pos = p1_pos;
            my_walls = p2_walls;
            opp_walls = p1_walls;
            my_goal_row = 0;
            opp_goal_row = 8;
        }

        // Helper to index into the flat output: plane * 81 + row * 9 + col
        auto idx = [](int plane, int r, int c) -> int {
            return plane * BOARD_SIZE * BOARD_SIZE + r * BOARD_SIZE + c;
        };

        // Plane 0: my pawn position (one-hot)
        output[idx(0, my_pos.row, my_pos.col)] = 1.0f;

        // Plane 1: opponent pawn position (one-hot)
        output[idx(1, opp_pos.row, opp_pos.col)] = 1.0f;

        // Plane 2: my walls remaining (normalized, broadcast)
        float my_walls_norm = static_cast<float>(my_walls) / 10.0f;
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                output[idx(2, r, c)] = my_walls_norm;

        // Plane 3: opponent walls remaining (normalized, broadcast)
        float opp_walls_norm = static_cast<float>(opp_walls) / 10.0f;
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                output[idx(3, r, c)] = opp_walls_norm;

        // Plane 4: horizontal walls (8x8 padded into 9x9)
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                if (h_walls[r][c])
                    output[idx(4, r, c)] = 1.0f;

        // Plane 5: vertical walls (8x8 padded into 9x9)
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                if (v_walls[r][c])
                    output[idx(5, r, c)] = 1.0f;

        // Planes 6-7: unused (zeros)

        // Plane 8: my goal row (full row set to 1)
        for (int c = 0; c < 9; c++)
            output[idx(8, my_goal_row, c)] = 1.0f;

        // Plane 9: opponent goal row (full row set to 1)
        for (int c = 0; c < 9; c++)
            output[idx(9, opp_goal_row, c)] = 1.0f;

        // Plane 10: legal pawn move positions
        std::vector<Move> pawn_moves = get_legal_pawn_moves();
        for (const auto& m : pawn_moves) {
            output[idx(10, m.row, m.col)] = 1.0f;
        }

        // Plane 11: am I player 1? (constant plane)
        float is_p1 = (current_player == 1) ? 1.0f : 0.0f;
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                output[idx(11, r, c)] = is_p1;
    }
};

} // namespace quoridor

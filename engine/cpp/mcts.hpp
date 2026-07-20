#pragma once
/**
 * mcts.hpp - Header-only batched MCTS for AlphaZero-style search.
 *
 * Key design: instead of calling the neural network once per leaf,
 * we accumulate leaf states into a batch. When the batch is full
 * (or all simulations are done), we return the batch to Python
 * for GPU evaluation, then expand nodes and backpropagate.
 *
 * This gives 100-1000x speedup:
 *   - Game state copy is ~200 bytes memcpy (vs Python deepcopy ~1ms)
 *   - No Python overhead in tree traversal
 *   - Batched GPU calls = good utilization
 */

#include "quoridor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace quoridor {

static constexpr float C_PUCT = 1.5f;
static constexpr float DIRICHLET_ALPHA = 0.3f;
static constexpr float DIRICHLET_EPSILON = 0.25f;

/**
 * Get proximity-filtered legal actions from the game engine.
 * Uses QuoridorGame::get_filtered_legal_actions() which applies the
 * proximity-based filter (Chebyshev distance, adjacent walls, edge walls).
 *
 * Much faster than the old BFS-per-wall approach because proximity checks
 * are O(1) per wall position.
 */
inline std::vector<Move> get_proximity_filtered_actions(const QuoridorGame& game) {
    return game.get_filtered_legal_actions();
}

struct MCTSNode {
    MCTSNode* parent = nullptr;
    int action = -1;         // action that led to this node
    float prior = 0.0f;      // P(s,a) from network

    int visit_count = 0;
    float total_value = 0.0f;

    bool is_expanded = false;
    QuoridorGame game_state;
    bool has_game_state = false;

    // Children stored as vector of unique_ptr for ownership
    std::vector<std::unique_ptr<MCTSNode>> children;

    MCTSNode() = default;
    MCTSNode(MCTSNode* p, int a, float pr)
        : parent(p), action(a), prior(pr) {}

    float q_value() const {
        if (visit_count == 0) return 0.0f;
        return total_value / static_cast<float>(visit_count);
    }

    float ucb_score(int parent_visits) const {
        float exploration = C_PUCT * prior * std::sqrt(static_cast<float>(parent_visits))
                           / (1.0f + static_cast<float>(visit_count));
        return q_value() + exploration;
    }

    MCTSNode* select_child() {
        float best_score = -1e9f;
        MCTSNode* best = nullptr;
        for (auto& child : children) {
            float score = child->ucb_score(visit_count);
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }
};

/**
 * Pending leaf evaluation: we traversed to this leaf and need the NN to evaluate it.
 */
struct PendingLeaf {
    MCTSNode* node;                     // the leaf node
    std::vector<MCTSNode*> search_path; // path from root to this leaf
};

/**
 * Batched MCTS search.
 *
 * Usage pattern from Python:
 *   1. Create MCTS instance
 *   2. Call search() which returns action probabilities
 *   3. search() internally calls the Python eval_fn for batched NN evaluation
 */
class MCTS {
public:
    int num_simulations;
    int batch_size;
    float temperature;
    bool add_noise;

    // Random engine
    std::mt19937 rng;

    MCTS(int num_sims = 100, int batch_sz = 8, unsigned seed = 0)
        : num_simulations(num_sims), batch_size(batch_sz),
          temperature(1.0f), add_noise(true) {
        if (seed == 0) {
            std::random_device rd;
            rng.seed(rd());
        } else {
            rng.seed(seed);
        }
    }

    /**
     * Run MCTS search and return action probabilities.
     *
     * eval_fn: a callable that takes a vector of encoded states (each 12*9*9 floats)
     *          and returns (policies, values) for the batch.
     *          - policies: vector of ACTION_SIZE floats per state
     *          - values: vector of single float per state
     *
     * This is the main entry point called from Python via pybind11.
     */
    template <typename EvalFn>
    std::array<float, ACTION_SIZE> search(const QuoridorGame& game, float temp,
                                          bool noise, EvalFn eval_fn) {
        temperature = temp;
        add_noise = noise;

        // Create root
        auto root = std::make_unique<MCTSNode>();
        root->game_state = game;
        root->has_game_state = true;
        root->is_expanded = true;

        // Evaluate root
        {
            std::vector<float> root_state(12 * 81);
            root->game_state.encode_state(root_state.data());

            std::vector<std::vector<float>> states_batch = {root_state};
            auto [policies, values] = eval_fn(states_batch);

            // Get proximity-filtered actions
            auto filtered = get_proximity_filtered_actions(root->game_state);

            // Build a mask: only filtered actions get nonzero prior
            std::vector<float> policy(ACTION_SIZE, 0.0f);
            float policy_sum = 0.0f;
            for (const auto& move : filtered) {
                int a = move.to_action();
                policy[a] = policies[0][a];
                policy_sum += policy[a];
            }
            if (policy_sum > 0.0f) {
                for (int i = 0; i < ACTION_SIZE; i++) policy[i] /= policy_sum;
            } else {
                // Uniform over filtered actions if NN gives all zeros
                float uniform = 1.0f / static_cast<float>(filtered.size());
                for (const auto& move : filtered) {
                    policy[move.to_action()] = uniform;
                }
            }

            // Split Dirichlet noise: 50% budget to pawn actions, 50% to wall actions
            if (add_noise) {
                // Separate filtered actions into pawn and wall
                std::vector<int> pawn_actions, wall_actions;
                for (const auto& move : filtered) {
                    int a = move.to_action();
                    if (move.type == MoveType::PAWN) {
                        pawn_actions.push_back(a);
                    } else {
                        wall_actions.push_back(a);
                    }
                }

                // Generate separate Dirichlet noise for each group
                std::vector<float> noise_combined(ACTION_SIZE, 0.0f);

                if (!pawn_actions.empty()) {
                    auto pawn_noise = dirichlet_noise(static_cast<int>(pawn_actions.size()));
                    for (size_t i = 0; i < pawn_actions.size(); i++) {
                        // 50% of total noise budget goes to pawn moves
                        noise_combined[pawn_actions[i]] = 0.5f * pawn_noise[i];
                    }
                }
                if (!wall_actions.empty()) {
                    auto wall_noise = dirichlet_noise(static_cast<int>(wall_actions.size()));
                    for (size_t i = 0; i < wall_actions.size(); i++) {
                        // 50% of total noise budget goes to wall moves
                        noise_combined[wall_actions[i]] = 0.5f * wall_noise[i];
                    }
                }
                // If one group is empty, the other gets all 100%
                if (pawn_actions.empty() && !wall_actions.empty()) {
                    for (int a : wall_actions) noise_combined[a] *= 2.0f;
                }
                if (wall_actions.empty() && !pawn_actions.empty()) {
                    for (int a : pawn_actions) noise_combined[a] *= 2.0f;
                }

                // Mix noise with policy
                policy_sum = 0.0f;
                for (int i = 0; i < ACTION_SIZE; i++) {
                    policy[i] = (1.0f - DIRICHLET_EPSILON) * policy[i]
                              + DIRICHLET_EPSILON * noise_combined[i];
                    policy_sum += policy[i];
                }
                if (policy_sum > 0.0f) {
                    for (int i = 0; i < ACTION_SIZE; i++) policy[i] /= policy_sum;
                }
            }

            // Create children from filtered moves with their priors
            root->children.reserve(filtered.size());
            for (const auto& move : filtered) {
                int action = move.to_action();
                auto child = std::make_unique<MCTSNode>(root.get(), action, policy[action]);
                root->children.push_back(std::move(child));
            }
        }

        // Run simulations in batches
        int sims_done = 0;
        while (sims_done < num_simulations) {
            std::vector<PendingLeaf> pending;
            pending.reserve(batch_size);

            // Collect a batch of leaves
            int batch_count = std::min(batch_size, num_simulations - sims_done);
            for (int b = 0; b < batch_count; b++) {
                MCTSNode* node = root.get();
                std::vector<MCTSNode*> search_path = {node};

                // Select: traverse tree to a leaf
                while (node->is_expanded && !node->children.empty()) {
                    node = node->select_child();
                    search_path.push_back(node);
                }

                // Ensure leaf has a game state
                if (!node->has_game_state) {
                    // Copy parent's state and apply the action
                    MCTSNode* parent_node = search_path[search_path.size() - 2];
                    node->game_state = parent_node->game_state;
                    Move move = Move::from_action(node->action);
                    node->game_state.play_move(move);
                    node->has_game_state = true;
                }

                // Check if terminal
                int winner = node->game_state.get_winner();
                if (winner != 0) {
                    // Terminal - backpropagate immediately
                    int parent_player = search_path[search_path.size() - 2]->game_state.current_player;
                    float value = (winner == parent_player) ? 1.0f : -1.0f;
                    backpropagate(search_path, value);
                    sims_done++;
                } else {
                    // Non-terminal leaf - add to batch for evaluation
                    pending.push_back({node, std::move(search_path)});
                }
            }

            if (pending.empty()) continue;

            // Encode all pending leaf states
            std::vector<std::vector<float>> states_batch;
            states_batch.reserve(pending.size());
            for (auto& leaf : pending) {
                std::vector<float> state(12 * 81);
                leaf.node->game_state.encode_state(state.data());
                states_batch.push_back(std::move(state));
            }

            // Call neural network for the batch
            auto [policies, values] = eval_fn(states_batch);

            // Expand nodes and backpropagate
            for (size_t i = 0; i < pending.size(); i++) {
                MCTSNode* node = pending[i].node;

                // Apply legal mask and normalize policy
                auto legal_mask = node->game_state.get_legal_action_mask();
                float policy_sum = 0.0f;
                for (int a = 0; a < ACTION_SIZE; a++) {
                    policies[i][a] *= legal_mask[a];
                    policy_sum += policies[i][a];
                }
                if (policy_sum > 0.0f) {
                    for (int a = 0; a < ACTION_SIZE; a++) {
                        policies[i][a] /= policy_sum;
                    }
                }

                // Create children — use proximity-filtered actions
                auto filtered = get_proximity_filtered_actions(node->game_state);

                // Zero out non-filtered actions and renormalize
                float filtered_sum = 0.0f;
                for (const auto& move : filtered) {
                    filtered_sum += policies[i][move.to_action()];
                }
                node->children.reserve(filtered.size());
                for (const auto& move : filtered) {
                    int action = move.to_action();
                    float prior = (filtered_sum > 0.0f) ? policies[i][action] / filtered_sum
                                                        : 1.0f / static_cast<float>(filtered.size());
                    auto child = std::make_unique<MCTSNode>(node, action, prior);
                    node->children.push_back(std::move(child));
                }
                node->is_expanded = true;

                // Backpropagate value (negate because value is from current player's perspective)
                float value = -values[i];
                backpropagate(pending[i].search_path, value);
                sims_done++;
            }
        }

        // Compute action probabilities from visit counts
        std::array<float, ACTION_SIZE> action_probs{};
        for (auto& child : root->children) {
            action_probs[child->action] = static_cast<float>(child->visit_count);
        }

        if (temperature <= 0.0f || temperature < 1e-6f) {
            // Greedy: pick the most visited
            int best_action = 0;
            float best_count = -1.0f;
            for (int i = 0; i < ACTION_SIZE; i++) {
                if (action_probs[i] > best_count) {
                    best_count = action_probs[i];
                    best_action = i;
                }
            }
            std::fill(action_probs.begin(), action_probs.end(), 0.0f);
            action_probs[best_action] = 1.0f;
        } else {
            // Apply temperature
            float sum = 0.0f;
            for (int i = 0; i < ACTION_SIZE; i++) {
                if (action_probs[i] > 0.0f) {
                    action_probs[i] = std::pow(action_probs[i], 1.0f / temperature);
                    sum += action_probs[i];
                }
            }
            if (sum > 0.0f) {
                for (int i = 0; i < ACTION_SIZE; i++) {
                    action_probs[i] /= sum;
                }
            }
        }

        return action_probs;
    }

private:
    void backpropagate(const std::vector<MCTSNode*>& search_path, float value) {
        // Value alternates sign as we go up (alternating player perspective)
        for (int i = static_cast<int>(search_path.size()) - 1; i >= 0; i--) {
            search_path[i]->visit_count++;
            search_path[i]->total_value += value;
            value = -value;
        }
    }

    std::vector<float> dirichlet_noise(int size) {
        // Generate Dirichlet noise using gamma distribution
        std::gamma_distribution<float> gamma(DIRICHLET_ALPHA, 1.0f);
        std::vector<float> noise(size);
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            noise[i] = gamma(rng);
            sum += noise[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < size; i++) noise[i] /= sum;
        }
        return noise;
    }
};

} // namespace quoridor

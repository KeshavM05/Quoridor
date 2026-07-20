/**
 * bindings.cpp - pybind11 module exposing C++ Quoridor engine and batched MCTS to Python.
 *
 * Module name: quoridor_cpp
 *
 * Exposed:
 *   - QuoridorGame class (positions, walls, play_move, get_legal_moves, get_winner)
 *   - Move class (type, row, col, to_action, from_action)
 *   - encode_state(game) -> numpy array (12, 9, 9)
 *   - get_legal_action_mask(game) -> numpy array (209,)
 *   - mcts_search(game, num_sims, batch_size, temperature, add_noise, eval_fn) -> numpy (209,)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "quoridor.hpp"
#include "mcts.hpp"

namespace py = pybind11;
using namespace quoridor;

/**
 * Encode a game state into a numpy array of shape (12, 9, 9).
 */
py::array_t<float> py_encode_state(const QuoridorGame& game) {
    std::vector<py::ssize_t> shape = {12, 9, 9};
    auto result = py::array_t<float>(shape);
    auto buf = result.mutable_unchecked<3>();

    // Get a pointer to the contiguous data
    float* ptr = static_cast<float*>(result.mutable_data());
    game.encode_state(ptr);

    return result;
}

/**
 * Get legal action mask as numpy array of shape (209,).
 */
py::array_t<float> py_get_legal_action_mask(const QuoridorGame& game) {
    auto mask = game.get_legal_action_mask();
    auto result = py::array_t<float>(ACTION_SIZE);
    float* ptr = static_cast<float*>(result.mutable_data());
    std::memcpy(ptr, mask.data(), ACTION_SIZE * sizeof(float));
    return result;
}

/**
 * MCTS search with batched neural network evaluation.
 *
 * eval_fn: Python callable that takes a numpy array of shape (batch_size, 12, 9, 9)
 *          and returns a tuple of (policies, values) where:
 *            policies: numpy array of shape (batch_size, 209)
 *            values: numpy array of shape (batch_size,) or (batch_size, 1)
 *
 * Returns: numpy array of shape (209,) with action probabilities.
 */
py::array_t<float> py_mcts_search(
    const QuoridorGame& game,
    int num_simulations,
    int batch_size,
    float temperature,
    bool add_noise,
    py::object eval_fn,
    unsigned seed = 0
) {
    if (eval_fn.is_none()) {
        throw std::runtime_error("eval_fn is required - must be a callable that evaluates game states");
    }
    MCTS mcts(num_simulations, batch_size, seed);

    // Wrap the Python eval function to match C++ interface
    py::function py_eval = eval_fn.cast<py::function>();
    auto cpp_eval_fn = [&py_eval](const std::vector<std::vector<float>>& states_batch)
        -> std::pair<std::vector<std::vector<float>>, std::vector<float>> {

        size_t batch_sz = states_batch.size();

        // Create numpy array of shape (batch_size, 12, 9, 9)
        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(batch_sz), 12, 9, 9};
        auto states_np = py::array_t<float>(shape);
        float* states_ptr = static_cast<float*>(states_np.mutable_data());
        for (size_t i = 0; i < batch_sz; i++) {
            std::memcpy(states_ptr + i * 12 * 81, states_batch[i].data(), 12 * 81 * sizeof(float));
        }

        // Call Python eval function
        py::tuple result = py_eval(states_np);

        // Parse policies: shape (batch_size, 209)
        py::array_t<float> policies_np = result[0].cast<py::array_t<float>>();
        auto policies_buf = policies_np.unchecked<2>();

        // Parse values: shape (batch_size,) or (batch_size, 1)
        py::array_t<float> values_np = result[1].cast<py::array_t<float>>();
        const float* values_ptr = static_cast<const float*>(values_np.data());
        int values_ndim = values_np.ndim();

        std::vector<std::vector<float>> policies(batch_sz, std::vector<float>(ACTION_SIZE));
        std::vector<float> values(batch_sz);

        for (size_t i = 0; i < batch_sz; i++) {
            for (int a = 0; a < ACTION_SIZE; a++) {
                policies[i][a] = policies_buf(i, a);
            }
            if (values_ndim == 1) {
                values[i] = values_ptr[i];
            } else {
                // Shape (batch_size, 1)
                values[i] = values_ptr[i];
            }
        }

        return {policies, values};
    };

    // Run the search
    auto action_probs = mcts.search(game, temperature, add_noise, cpp_eval_fn);

    // Convert to numpy
    auto result = py::array_t<float>(ACTION_SIZE);
    float* result_ptr = static_cast<float*>(result.mutable_data());
    std::memcpy(result_ptr, action_probs.data(), ACTION_SIZE * sizeof(float));
    return result;
}

PYBIND11_MODULE(quoridor_cpp, m) {
    m.doc() = "C++ Quoridor engine with batched MCTS (100-1000x faster than Python)";

    // Expose MoveType enum
    py::enum_<MoveType>(m, "MoveType")
        .value("PAWN", MoveType::PAWN)
        .value("WALL_H", MoveType::WALL_H)
        .value("WALL_V", MoveType::WALL_V);

    // Expose Move struct
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<MoveType, int, int>())
        .def_readwrite("type", &Move::type)
        .def_readwrite("row", &Move::row)
        .def_readwrite("col", &Move::col)
        .def("to_action", &Move::to_action)
        .def_static("from_action", &Move::from_action)
        .def("__eq__", &Move::operator==)
        .def("__repr__", [](const Move& m) {
            const char* type_str = (m.type == MoveType::PAWN) ? "PAWN" :
                                   (m.type == MoveType::WALL_H) ? "WALL_H" : "WALL_V";
            return std::string("Move(") + type_str + ", " +
                   std::to_string(m.row) + ", " + std::to_string(m.col) + ")";
        });

    // Expose QuoridorGame class
    py::class_<QuoridorGame>(m, "QuoridorGame")
        .def(py::init<>())
        .def(py::init<const QuoridorGame&>()) // Copy constructor
        .def("reset", &QuoridorGame::reset)
        .def("play_move", &QuoridorGame::play_move)
        .def("get_legal_moves", &QuoridorGame::get_legal_moves)
        .def("get_legal_pawn_moves", &QuoridorGame::get_legal_pawn_moves)
        .def("get_legal_wall_moves", &QuoridorGame::get_legal_wall_moves)
        .def("get_filtered_legal_actions", &QuoridorGame::get_filtered_legal_actions)
        .def("get_winner", &QuoridorGame::get_winner)
        .def("is_blocked", &QuoridorGame::is_blocked)
        .def("path_exists", [](const QuoridorGame& g, py::tuple pos, int target_row) {
            return g.path_exists(Position(pos[0].cast<int>(), pos[1].cast<int>()), target_row);
        })
        .def_property("p1_pos",
            [](const QuoridorGame& g) { return py::make_tuple(int(g.p1_pos.row), int(g.p1_pos.col)); },
            [](QuoridorGame& g, py::tuple t) { g.p1_pos = Position(t[0].cast<int>(), t[1].cast<int>()); })
        .def_property("p2_pos",
            [](const QuoridorGame& g) { return py::make_tuple(int(g.p2_pos.row), int(g.p2_pos.col)); },
            [](QuoridorGame& g, py::tuple t) { g.p2_pos = Position(t[0].cast<int>(), t[1].cast<int>()); })
        .def_readwrite("p1_walls", &QuoridorGame::p1_walls)
        .def_readwrite("p2_walls", &QuoridorGame::p2_walls)
        .def_readwrite("current_player", &QuoridorGame::current_player)
        .def_property_readonly("h_walls", [](const QuoridorGame& g) {
            auto result = py::array_t<bool>({8, 8});
            bool* ptr = static_cast<bool*>(result.mutable_data());
            std::memcpy(ptr, g.h_walls, 64 * sizeof(bool));
            return result;
        })
        .def_property_readonly("v_walls", [](const QuoridorGame& g) {
            auto result = py::array_t<bool>({8, 8});
            bool* ptr = static_cast<bool*>(result.mutable_data());
            std::memcpy(ptr, g.v_walls, 64 * sizeof(bool));
            return result;
        })
        .def("set_h_wall", [](QuoridorGame& g, int r, int c, bool val) { g.h_walls[r][c] = val; })
        .def("set_v_wall", [](QuoridorGame& g, int r, int c, bool val) { g.v_walls[r][c] = val; })
        .def("__copy__", [](const QuoridorGame& g) { return QuoridorGame(g); })
        .def("__deepcopy__", [](const QuoridorGame& g, py::dict) { return QuoridorGame(g); })
        .def("__repr__", [](const QuoridorGame& g) {
            return std::string("<QuoridorGame p1=(" + std::to_string(g.p1_pos.row) + "," +
                   std::to_string(g.p1_pos.col) + ") p2=(" + std::to_string(g.p2_pos.row) + "," +
                   std::to_string(g.p2_pos.col) + ") turn=P" + std::to_string(g.current_player) + ">");
        });

    // Expose encode_state as a module-level function
    m.def("encode_state", &py_encode_state,
          "Encode game state into numpy array of shape (12, 9, 9)",
          py::arg("game"));

    // Expose get_legal_action_mask
    m.def("get_legal_action_mask", &py_get_legal_action_mask,
          "Get binary mask of legal actions, shape (209,)",
          py::arg("game"));

    // Expose mcts_search
    m.def("mcts_search", &py_mcts_search,
          R"(Run batched MCTS search.

Args:
    game: QuoridorGame state to search from
    num_sims: number of MCTS simulations (default 100)
    batch_size: how many leaves to batch for NN eval (default 8)
    temperature: temperature for action selection (1.0 = proportional, ~0 = greedy)
    add_noise: whether to add Dirichlet noise at root
    eval_fn: Python callable: numpy(batch, 12, 9, 9) -> (policies(batch, 209), values(batch,))
    seed: random seed (0 = random)

Returns:
    numpy array of shape (209,) with action probabilities
)",
          py::arg("game"),
          py::arg("num_sims") = 100,
          py::arg("batch_size") = 8,
          py::arg("temperature") = 1.0f,
          py::arg("add_noise") = true,
          py::arg("eval_fn"),
          py::arg("seed") = 0);

    // Module constants
    m.attr("ACTION_SIZE") = ACTION_SIZE;
    m.attr("BOARD_SIZE") = BOARD_SIZE;
}

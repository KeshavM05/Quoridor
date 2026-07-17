"""
AlphaZero-style ResNet for Quoridor.

Input: 9x9x12 tensor encoding board state
Output: policy (probability over all moves) + value ([-1, 1])

Board encoding (12 planes):
  0: current player pawn position (one-hot 9x9)
  1: opponent pawn position (one-hot 9x9)
  2: current player walls remaining (scalar broadcast)
  3: opponent walls remaining (scalar broadcast)
  4-5: horizontal walls placed (8x8 padded to 9x9, split by player)
  6-7: vertical walls placed (8x8 padded to 9x9, split by player)
  8: current player's goal row (binary mask)
  9: opponent's goal row (binary mask)
  10: legal move mask (for cells reachable by pawn)
  11: constant plane = 1 if current player is P1, 0 if P2

Action space (total = 81 + 128 = 209):
  - Pawn moves: 9x9 = 81 positions
  - H walls: 8x8 = 64 positions
  - V walls: 8x8 = 64 positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 9
NUM_CHANNELS = 12
ACTION_SIZE = 81 + 64 + 64  # 209

NUM_RES_BLOCKS = 12
HIDDEN_CHANNELS = 256


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class QuoridorNet(nn.Module):
    def __init__(self, num_res_blocks=NUM_RES_BLOCKS, channels=HIDDEN_CHANNELS):
        super().__init__()
        self.conv_input = nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def encode_state(game, canonical_player=None):
    """
    Encode a QuoridorGame state into a 12x9x9 tensor.
    If canonical_player is None, uses game.current_player.
    The board is always from the perspective of the current player.
    """
    import numpy as np

    if canonical_player is None:
        canonical_player = game.current_player

    if canonical_player == 1:
        my_pos = game.p1_pos
        opp_pos = game.p2_pos
        my_walls = game.p1_walls
        opp_walls = game.p2_walls
        my_goal_row = 8
        opp_goal_row = 0
    else:
        my_pos = game.p2_pos
        opp_pos = game.p1_pos
        my_walls = game.p2_walls
        opp_walls = game.p1_walls
        my_goal_row = 0
        opp_goal_row = 8

    planes = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Plane 0: my pawn
    planes[0, my_pos[0], my_pos[1]] = 1.0
    # Plane 1: opponent pawn
    planes[1, opp_pos[0], opp_pos[1]] = 1.0
    # Plane 2: my walls remaining (normalized)
    planes[2] = my_walls / 10.0
    # Plane 3: opp walls remaining
    planes[3] = opp_walls / 10.0
    # Plane 4: horizontal walls (padded)
    planes[4, :8, :8] = game.h_walls.astype(np.float32)
    # Plane 5: vertical walls (padded)
    planes[5, :8, :8] = game.v_walls.astype(np.float32)
    # Plane 6-7: unused (could split walls by who placed them, but we don't track that)
    # Plane 8: my goal row
    planes[8, my_goal_row, :] = 1.0
    # Plane 9: opponent goal row
    planes[9, opp_goal_row, :] = 1.0
    # Plane 10: legal pawn move positions
    legal = game.get_legal_moves()
    for m in legal:
        if m[0] == 'move':
            planes[10, m[1], m[2]] = 1.0
    # Plane 11: am I player 1?
    planes[11] = 1.0 if canonical_player == 1 else 0.0

    return planes


def move_to_action(move):
    """Convert a game move tuple to an action index (0-208)."""
    if move[0] == 'move':
        return move[1] * BOARD_SIZE + move[2]
    elif move[0] == 'wall':
        orient, r, c = move[1], move[2], move[3]
        if orient == 'h':
            return 81 + r * 8 + c
        else:
            return 81 + 64 + r * 8 + c
    raise ValueError(f"Unknown move: {move}")


def action_to_move(action):
    """Convert an action index back to a game move tuple."""
    if action < 81:
        r = action // BOARD_SIZE
        c = action % BOARD_SIZE
        return ('move', r, c)
    elif action < 81 + 64:
        idx = action - 81
        r = idx // 8
        c = idx % 8
        return ('wall', 'h', r, c)
    else:
        idx = action - 81 - 64
        r = idx // 8
        c = idx % 8
        return ('wall', 'v', r, c)


def get_legal_action_mask(game):
    """Return a binary mask over the action space for legal moves."""
    import numpy as np
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move in game.get_legal_moves():
        mask[move_to_action(move)] = 1.0
    return mask

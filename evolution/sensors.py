import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.logic import GameLogic
from game.constants import Direction, FieldType, FIELDSIZE, DIR_DELTA, FOOD_STEPS, BOMB_STEPS

# -------------------------------------------------------
# Raw-input constants
# -------------------------------------------------------

# Normalized cell values for the raw grid encoding
_RAW_CELL_FREE     = 0.0
_RAW_CELL_WALL     = 1 / 6
_RAW_CELL_EXPLODED = 2 / 6
_RAW_CELL_FOOD     = 3 / 6
_RAW_CELL_BOMB     = 4 / 6
_RAW_CELL_BODY     = 5 / 6
_RAW_CELL_HEAD     = 1.0

_DIR_ONEHOT = {
    Direction.RIGHT : [1.0, 0.0, 0.0, 0.0],
    Direction.UP    : [0.0, 1.0, 0.0, 0.0],
    Direction.LEFT  : [0.0, 0.0, 1.0, 0.0],
    Direction.DOWN  : [0.0, 0.0, 0.0, 1.0],
}

# 100 grid cells + 4 direction one-hot + food_timer + bomb_timer + explosion + snake_length
RAW_INPUT_SIZE = FIELDSIZE * FIELDSIZE + 4 + 4


# -------------------------------------------------------
# Lookup tables: current direction → absolute direction after a relative turn
# -------------------------------------------------------

TURN_LEFT = {
    Direction.RIGHT : Direction.UP,
    Direction.UP    : Direction.LEFT,
    Direction.LEFT  : Direction.DOWN,
    Direction.DOWN  : Direction.RIGHT,
}

TURN_RIGHT = {
    Direction.RIGHT : Direction.DOWN,
    Direction.DOWN  : Direction.LEFT,
    Direction.LEFT  : Direction.UP,
    Direction.UP    : Direction.RIGHT,
}


# -------------------------------------------------------
# MoveHelper
# -------------------------------------------------------

class MoveHelper :
    """Computes relative sensor inputs for the neural network from the current game state."""

    FORWARD  = "forward"
    LEFT     = "left"
    RIGHT    = "right"
    BACKWARD = "backward"

    def __init__(self, game: GameLogic, profile: dict = None) :
        self.game    = game
        self.profile = profile  # None → all 14 sensors at weight 1.0


    # --- Public sensors ----------------------------------------------------

    def can_move(self, direction: str) -> bool :
        """True if moving in this relative direction is safe (no wall/explosion + no body collision)."""
        head = self.game.snake[0]

        if direction == self.FORWARD :
            abs_dir = self.game.direction
        elif direction == self.LEFT :
            abs_dir = TURN_LEFT[self.game.direction]
        else :
            abs_dir = TURN_RIGHT[self.game.direction]

        nx, ny = self._step(abs_dir, head)

        if self.game.grid[ny][nx] != FieldType.FREE :
            return False

        for segment in self.game.snake :
            if nx == segment[0] and ny == segment[1] :
                return False

        return True

    def is_food(self, direction: str) -> float :
        """True if food lies anywhere in this relative direction (not just the adjacent cell)."""
        head = self.game.snake[0]

        if direction == self.FORWARD  : return self._is_food_forward(head)
        if direction == self.LEFT     : return self._is_food_left(head)
        if direction == self.RIGHT    : return self._is_food_right(head)
        if direction == self.BACKWARD : return self._is_food_backward(head)

    def get_inputs(self) -> list :
        """Return the input vector for the network, filtered and weighted by the active profile."""
        if self.profile is None :
            return [v * 1.0 for v in self._all_raw_values()]
        if "_raw" in self.profile :
            return self.get_raw_inputs()
        return [
            SENSOR_FUNCS[name](self) * weight
            for name, weight in self.profile.items()
            if weight > 0
        ]

    def get_raw_inputs(self) -> list :
        """Return the full game state as a flat normalized vector (RAW_INPUT_SIZE values).

        Layout:
          [0..99]   10×10 grid, row-major (y=0..9, x=0..9), normalized cell values
          [100..103] direction one-hot: RIGHT / UP / LEFT / DOWN
          [104]     food_timer  / FOOD_STEPS
          [105]     bomb_timer  / BOMB_STEPS  (0 if no bomb)
          [106]     explosion_active  (0 or 1)
          [107]     snake_length / (FIELDSIZE²)
        """
        game     = self.game
        head     = game.snake[0] if game.snake else (-1, -1)
        body_set = set((x, y) for x, y in game.snake[1:])
        fx, fy   = game.food_pos
        bx, by   = game.bomb_pos if (game.bomb and game.bomb_pos) else (None, None)

        inputs = []
        for y in range(FIELDSIZE) :
            for x in range(FIELDSIZE) :
                ft = game.grid[y][x]
                if ft == FieldType.WALL :
                    val = _RAW_CELL_WALL
                elif ft == FieldType.EXPLODED :
                    val = _RAW_CELL_EXPLODED
                elif (x, y) == head :
                    val = _RAW_CELL_HEAD
                elif (x, y) in body_set :
                    val = _RAW_CELL_BODY
                elif x == fx and y == fy :
                    val = _RAW_CELL_FOOD
                elif bx is not None and x == bx and y == by :
                    val = _RAW_CELL_BOMB
                else :
                    val = _RAW_CELL_FREE
                inputs.append(val)

        inputs.extend(_DIR_ONEHOT[game.direction])
        inputs.append(game.food_timer / FOOD_STEPS)
        inputs.append((game.bomb_timer / BOMB_STEPS) if game.bomb else 0.0)
        inputs.append(1.0 if game.explosion else 0.0)
        inputs.append(len(game.snake) / (FIELDSIZE * FIELDSIZE))

        return inputs

    def _all_raw_values(self) -> list :
        """All 19 sensor values at weight 1.0 (used when no profile is set)."""
        return [func(self) for func in SENSOR_FUNCS.values()]

    def food_timer_normalized(self) -> float :
        """How close food is to becoming a bomb (0.0 = just spawned, 1.0 = about to explode)."""
        return self.game.food_timer / FOOD_STEPS

    def bomb_timer_normalized(self) -> float :
        """How close the active bomb is to exploding (0.0 = just spawned, 1.0 = about to explode). 0 if no bomb."""
        if not self.game.bomb :
            return 0.0
        return self.game.bomb_timer / BOMB_STEPS

    def explosion_active(self) -> float :
        """1.0 if there are active explosion cells on the grid, 0.0 otherwise."""
        return 1.0 if self.game.explosion else 0.0

    def food_distance_normalized(self) -> float :
        """Manhattan distance from head to food, normalized to [0, 1]."""
        hx, hy = self.game.snake[0]
        fx, fy = self.game.food_pos
        max_dist = (FIELDSIZE - 1) * 2
        return (abs(fx - hx) + abs(fy - hy)) / max_dist

    def snake_length_normalized(self) -> float :
        """Snake length normalized to [0, 1] relative to total grid size."""
        return len(self.game.snake) / (FIELDSIZE * FIELDSIZE)

    def is_bomb(self, direction: str) -> float :
        """0.0–1.0: how close an active bomb is in this relative direction (0 = none/far)."""
        if not self.game.bomb or self.game.bomb_pos is None :
            return 0.0

        head = self.game.snake[0]

        if direction == self.FORWARD  : return self._is_bomb_forward(head)
        if direction == self.LEFT     : return self._is_bomb_left(head)
        if direction == self.RIGHT    : return self._is_bomb_right(head)
        if direction == self.BACKWARD : return self._is_bomb_backward(head)

    def body_proximity(self, direction: str) -> float :
        """0.0–1.0: proximity of the nearest body segment along a ray in this relative direction.
        0.9 = body 1 cell away (imminent trap), 0.1 = 9 cells away, 0.0 = no body on that axis."""
        if direction == self.FORWARD :
            abs_dir = self.game.direction
        elif direction == self.LEFT :
            abs_dir = TURN_LEFT[self.game.direction]
        else :
            abs_dir = TURN_RIGHT[self.game.direction]

        body_set = set((s[0], s[1]) for s in self.game.snake[1:])
        dx, dy   = DIR_DELTA[abs_dir]
        x, y     = self.game.snake[0]
        for dist in range(1, FIELDSIZE) :
            x = (x + dx) % FIELDSIZE
            y = (y + dy) % FIELDSIZE
            if (x, y) in body_set :
                return 1.0 - dist / FIELDSIZE
        return 0.0


# --- Food direction checks: continuous proximity (0.0 = not in this dir, 1.0 = adjacent) ---
    # y=0 is the bottom row (OpenGL convention), so UP means food_y > head_y
    # Formula: 1 - dist/FIELDSIZE  (dist=1 → 0.9,  dist=9 → 0.1,  wrong direction → 0.0)

    def _is_food_forward(self, head: tuple) -> float :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = fx - hx if fx > hx else 0
        elif d == Direction.LEFT  : dist = hx - fx if fx < hx else 0
        elif d == Direction.UP    : dist = fy - hy if fy > hy else 0
        else                      : dist = hy - fy if fy < hy else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0

    def _is_food_left(self, head: tuple) -> float :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = fy - hy if fy > hy else 0
        elif d == Direction.UP    : dist = hx - fx if fx < hx else 0
        elif d == Direction.LEFT  : dist = hy - fy if fy < hy else 0
        else                      : dist = fx - hx if fx > hx else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0

    def _is_food_right(self, head: tuple) -> float :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = hy - fy if fy < hy else 0
        elif d == Direction.DOWN  : dist = hx - fx if fx < hx else 0
        elif d == Direction.LEFT  : dist = fy - hy if fy > hy else 0
        else                      : dist = fx - hx if fx > hx else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0

    def _is_food_backward(self, head: tuple) -> float :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = hx - fx if fx < hx else 0
        elif d == Direction.LEFT  : dist = fx - hx if fx > hx else 0
        elif d == Direction.UP    : dist = hy - fy if fy < hy else 0
        else                      : dist = fy - hy if fy > hy else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0


    # --- Bomb direction checks: continuous proximity (same scheme as food) ---

    def _is_bomb_forward(self, head: tuple) -> float :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = bx - hx if bx > hx else 0
        elif d == Direction.LEFT  : dist = hx - bx if bx < hx else 0
        elif d == Direction.UP    : dist = by - hy if by > hy else 0
        else                      : dist = hy - by if by < hy else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0

    def _is_bomb_left(self, head: tuple) -> float :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = by - hy if by > hy else 0
        elif d == Direction.UP    : dist = hx - bx if bx < hx else 0
        elif d == Direction.LEFT  : dist = hy - by if by < hy else 0
        else                      : dist = bx - hx if bx > hx else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0

    def _is_bomb_right(self, head: tuple) -> float :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = hy - by if by < hy else 0
        elif d == Direction.DOWN  : dist = hx - bx if bx < hx else 0
        elif d == Direction.LEFT  : dist = by - hy if by > hy else 0
        else                      : dist = bx - hx if bx > hx else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0

    def _is_bomb_backward(self, head: tuple) -> float :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if   d == Direction.RIGHT : dist = hx - bx if bx < hx else 0
        elif d == Direction.LEFT  : dist = bx - hx if bx > hx else 0
        elif d == Direction.UP    : dist = hy - by if by < hy else 0
        else                      : dist = by - hy if by > hy else 0
        return (1.0 - dist / FIELDSIZE) if dist > 0 else 0.0


    # --- Shared helpers ----------------------------------------------------

    def _step(self, direction: Direction, pos: tuple) -> tuple :
        """Return the cell one step in the given absolute direction (with wrap-around)."""
        dx, dy = DIR_DELTA[direction]
        x,  y  = pos
        return ((x + dx) % FIELDSIZE, (y + dy) % FIELDSIZE)


# Ordered mapping from sensor name to callable — must match profiles.py key order.
SENSOR_FUNCS: dict = {
    "can_move_forward" : lambda h: 1 if h.can_move(h.FORWARD) else 0,
    "can_move_left"    : lambda h: 1 if h.can_move(h.LEFT)    else 0,
    "can_move_right"   : lambda h: 1 if h.can_move(h.RIGHT)   else 0,
    "is_food_forward"  : lambda h: h.is_food(h.FORWARD),
    "is_food_left"     : lambda h: h.is_food(h.LEFT),
    "is_food_right"    : lambda h: h.is_food(h.RIGHT),
    "is_bomb_forward"  : lambda h: h.is_bomb(h.FORWARD),
    "is_bomb_left"     : lambda h: h.is_bomb(h.LEFT),
    "is_bomb_right"    : lambda h: h.is_bomb(h.RIGHT),
    "food_timer"       : lambda h: h.food_timer_normalized(),
    "bomb_timer"       : lambda h: h.bomb_timer_normalized(),
    "explosion_active" : lambda h: h.explosion_active(),
    "food_distance"    : lambda h: h.food_distance_normalized(),
    "snake_length"     : lambda h: h.snake_length_normalized(),
    "body_forward"     : lambda h: h.body_proximity(h.FORWARD),
    "body_left"        : lambda h: h.body_proximity(h.LEFT),
    "body_right"       : lambda h: h.body_proximity(h.RIGHT),
    "is_food_backward" : lambda h: h.is_food(h.BACKWARD),
    "is_bomb_backward" : lambda h: h.is_bomb(h.BACKWARD),
}


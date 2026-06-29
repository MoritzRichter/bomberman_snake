import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.logic import GameLogic
from game.constants import Direction, FieldType, FIELDSIZE, DIR_DELTA, FOOD_STEPS, BOMB_STEPS

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

    FORWARD = "forward"
    LEFT    = "left"
    RIGHT   = "right"

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

        if direction == self.FORWARD :
            return self._is_food_forward(head)
        if direction == self.LEFT :
            return self._is_food_left(head)
        if direction == self.RIGHT :
            return self._is_food_right(head)

    def get_inputs(self) -> list :
        """Return the input vector for the network, filtered and weighted by the active profile."""
        if self.profile is None :
            return [v * 1.0 for v in self._all_raw_values()]
        return [
            SENSOR_FUNCS[name](self) * weight
            for name, weight in self.profile.items()
            if weight > 0
        ]

    def _all_raw_values(self) -> list :
        """All 14 sensor values at weight 1.0 (used when no profile is set)."""
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

        if direction == self.FORWARD :
            return self._is_bomb_forward(head)
        if direction == self.LEFT :
            return self._is_bomb_left(head)
        if direction == self.RIGHT :
            return self._is_bomb_right(head)

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
}


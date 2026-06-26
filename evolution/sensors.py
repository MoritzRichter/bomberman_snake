import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.logic import GameLogic
from game.constants import Direction, FieldType, FIELDSIZE, DIR_DELTA

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

    def __init__(self, game: GameLogic) :
        self.game = game


    # --- Public sensors ----------------------------------------------------

    def can_move(self, direction: str) -> bool :
        """True if moving in this relative direction is safe (no wall/explosion + no body collision)."""
        head = self.game.snake[0]
        can  = True

        if direction == self.FORWARD :
            can = self._can_move_forward(head)
        elif direction == self.LEFT :
            can = self._can_move_left(head)
        elif direction == self.RIGHT :
            can = self._can_move_right(head)

        # Body collision: check whether the next forward step lands on any segment
        for segment in self.game.snake :
            if self._is_collision(head, segment) :
                can = False

        return can

    def is_food(self, direction: str) -> bool :
        """True if food lies anywhere in this relative direction (not just the adjacent cell)."""
        head = self.game.snake[0]

        if direction == self.FORWARD :
            return self._is_food_forward(head)
        if direction == self.LEFT :
            return self._is_food_left(head)
        if direction == self.RIGHT :
            return self._is_food_right(head)

    def get_inputs(self) -> list :
        """Assemble the full input vector to feed into the neural network."""
        return [
            1 if self.can_move(self.FORWARD) else 0,
            1 if self.can_move(self.LEFT)    else 0,
            1 if self.can_move(self.RIGHT)   else 0,
            1 if self.is_food(self.FORWARD)  else 0,
            1 if self.is_food(self.LEFT)     else 0,
            1 if self.is_food(self.RIGHT)    else 0,
            1 if self.is_bomb(self.FORWARD)  else 0,
            1 if self.is_bomb(self.LEFT)     else 0,
            1 if self.is_bomb(self.RIGHT)    else 0,
        ]

    def is_bomb(self, direction: str) -> bool :
        """True if an active bomb lies anywhere in this relative direction."""
        if not self.game.bomb or self.game.bomb_pos is None :
            return False

        head = self.game.snake[0]

        if direction == self.FORWARD :
            return self._is_bomb_forward(head)
        if direction == self.LEFT :
            return self._is_bomb_left(head)
        if direction == self.RIGHT :
            return self._is_bomb_right(head)


    # --- Wall / grid checks ------------------------------------------------

    def _can_move_forward(self, head: tuple) -> bool :
        nx, ny = self._step(self.game.direction, head)
        return self.game.grid[ny][nx] == FieldType.FREE

    def _can_move_left(self, head: tuple) -> bool :
        nx, ny = self._step(TURN_LEFT[self.game.direction], head)
        return self.game.grid[ny][nx] == FieldType.FREE

    def _can_move_right(self, head: tuple) -> bool :
        nx, ny = self._step(TURN_RIGHT[self.game.direction], head)
        return self.game.grid[ny][nx] == FieldType.FREE


    # --- Food direction checks (range, not adjacent) -----------------------
    # y=0 is the bottom row (OpenGL convention), so UP means food_y > head_y

    def _is_food_forward(self, head: tuple) -> bool :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if d == Direction.RIGHT : return fx > hx
        if d == Direction.LEFT  : return fx < hx
        if d == Direction.UP    : return fy > hy
        if d == Direction.DOWN  : return fy < hy

    def _is_food_left(self, head: tuple) -> bool :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if d == Direction.RIGHT : return fy > hy
        if d == Direction.UP    : return fx < hx
        if d == Direction.LEFT  : return fy < hy
        if d == Direction.DOWN  : return fx > hx

    def _is_food_right(self, head: tuple) -> bool :
        hx, hy = head
        fx, fy = self.game.food_pos
        d = self.game.direction
        if d == Direction.RIGHT : return fy < hy
        if d == Direction.DOWN  : return fx < hx
        if d == Direction.LEFT  : return fy > hy
        if d == Direction.UP    : return fx > hx


    # --- Bomb direction checks (range, not adjacent) -----------------------

    def _is_bomb_forward(self, head: tuple) -> bool :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if d == Direction.RIGHT : return bx > hx
        if d == Direction.LEFT  : return bx < hx
        if d == Direction.UP    : return by > hy
        if d == Direction.DOWN  : return by < hy

    def _is_bomb_left(self, head: tuple) -> bool :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if d == Direction.RIGHT : return by > hy
        if d == Direction.UP    : return bx < hx
        if d == Direction.LEFT  : return by < hy
        if d == Direction.DOWN  : return bx > hx

    def _is_bomb_right(self, head: tuple) -> bool :
        hx, hy = head
        bx, by = self.game.bomb_pos
        d = self.game.direction
        if d == Direction.RIGHT : return by < hy
        if d == Direction.DOWN  : return bx < hx
        if d == Direction.LEFT  : return by > hy
        if d == Direction.UP    : return bx > hx


    # --- Body collision ----------------------------------------------------

    def _is_collision(self, head: tuple, segment: tuple) -> bool :
        """True if the snake's next forward step lands on this segment."""
        nx, ny = self._step(self.game.direction, head)
        return nx == segment[0] and ny == segment[1]


    # --- Shared helpers ----------------------------------------------------

    def _step(self, direction: Direction, pos: tuple) -> tuple :
        """Return the cell one step in the given absolute direction (with wrap-around)."""
        dx, dy = DIR_DELTA[direction]
        x,  y  = pos
        return ((x + dx) % FIELDSIZE, (y + dy) % FIELDSIZE)


from enum import IntEnum

FIELDSIZE = 10
EXPLOSION_REACH = 3

# Discrete step counts ported from the timing constants in logic.c
FOOD_STEPS = 6       # steps until uneaten food spawns a bomb
BOMB_STEPS = 3       # steps until bomb explodes
EXPLOSION_STEPS = 2  # steps until explosion disappears

START_X = 6
START_Y = 6


# Terrain types stored in the game grid (logic layer only)
class FieldType(IntEnum):
    FREE = 0
    WALL = 1
    EXPLODED = 2


class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


# Observation values used by the renderer — superset of FieldType,
# also covers dynamic entities (snake, food, bomb) not stored in the grid
class Cell(IntEnum):
    FREE = 0
    WALL = 1
    EXPLODED = 2
    SNAKE_BODY = 3
    SNAKE_HEAD = 4
    FOOD = 5
    BOMB = 6


OPPOSITE = {
    Direction.LEFT:  Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
    Direction.UP:    Direction.DOWN,
    Direction.DOWN:  Direction.UP,
}

# y=+1 means UP — matches the OpenGL convention used in the original C code
DIR_DELTA = {
    Direction.LEFT:  (-1,  0),
    Direction.RIGHT: ( 1,  0),
    Direction.UP:    ( 0,  1),
    Direction.DOWN:  ( 0, -1),
}

# Directions an explosion's arms travel from its centre.
# "plus" = classic + shape (orthogonal), "x" = diagonal X shape.
EXPLOSION_SHAPES: dict[str, list[tuple[int, int]]] = {
    "plus": [( 1,  0), (-1,  0), ( 0,  1), ( 0, -1)],
    "x":    [( 1,  1), ( 1, -1), (-1,  1), (-1, -1)],
}

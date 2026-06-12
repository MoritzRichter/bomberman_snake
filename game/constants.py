from enum import IntEnum

FIELDSIZE = 10
EXPLOSION_REACH = 3

FOOD_STEPS = 6
BOMB_STEPS = 3
EXPLOSION_STEPS = 2

START_X = 6
START_Y = 6


class FieldType(IntEnum):
    FREE = 0
    WALL = 1
    EXPLODED = 2


class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


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

DIR_DELTA = {
    Direction.LEFT:  (-1,  0),
    Direction.RIGHT: ( 1,  0),
    Direction.UP:    ( 0,  1),
    Direction.DOWN:  ( 0, -1),
}

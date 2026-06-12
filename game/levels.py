from .constants import FieldType

W = FieldType.WALL
F = FieldType.FREE

# grid[y][x], y=0 = unterste Reihe (wie im C-Original: g_level[y][x])
# Portiert aus cg_base/src/level.c

LEVEL_1 = [
    [W, W, W, W, W, W, W, W, W, W],  # y=0
    [W, F, F, F, F, F, F, F, F, W],  # y=1
    [W, F, F, F, F, F, F, F, F, W],  # y=2
    [W, F, F, F, F, F, F, F, F, W],  # y=3
    [W, F, F, F, F, F, F, F, F, W],  # y=4
    [W, F, F, F, F, F, F, F, F, W],  # y=5
    [W, F, F, F, F, F, F, F, F, W],  # y=6  ← Startpositionen (4,6)(5,6)(6,6)
    [W, F, F, F, F, F, F, F, F, W],  # y=7
    [W, F, F, F, F, F, F, F, F, W],  # y=8
    [W, W, W, W, W, W, W, W, W, W],  # y=9
]

LEVEL_2 = [
    [W, W, W, W, F, F, W, W, W, W],  # y=0
    [W, F, F, F, F, F, F, F, F, W],  # y=1
    [W, F, F, F, F, F, F, F, F, W],  # y=2
    [W, F, F, F, F, F, F, F, F, W],  # y=3
    [F, F, F, F, W, W, F, F, F, F],  # y=4
    [F, F, F, F, W, W, F, F, F, F],  # y=5
    [W, F, F, F, F, F, F, F, F, W],  # y=6  ← Startpositionen frei
    [W, F, F, F, F, F, F, F, F, W],  # y=7
    [W, F, F, F, F, F, F, F, F, W],  # y=8
    [W, W, W, W, F, F, W, W, W, W],  # y=9
]

LEVEL_3 = [
    [W, F, W, F, W, W, F, W, F, W],  # y=0
    [F, F, F, F, F, F, F, F, F, F],  # y=1
    [W, F, W, F, W, W, F, W, F, W],  # y=2
    [F, F, F, F, F, F, F, F, F, F],  # y=3
    [W, F, W, F, W, W, F, W, F, W],  # y=4
    [W, F, W, F, W, W, F, W, F, W],  # y=5
    [F, F, F, F, F, F, F, F, F, F],  # y=6  ← Startpositionen frei
    [W, F, W, F, W, W, F, W, F, W],  # y=7
    [F, F, F, F, F, F, F, F, F, F],  # y=8
    [W, F, W, F, W, W, F, W, F, W],  # y=9
]

LEVELS = {1: LEVEL_1, 2: LEVEL_2, 3: LEVEL_3}

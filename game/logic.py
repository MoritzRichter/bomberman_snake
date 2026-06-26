import random
from .constants import (
    FIELDSIZE, EXPLOSION_REACH, FOOD_STEPS, BOMB_STEPS, EXPLOSION_STEPS,
    Direction, FieldType, OPPOSITE, DIR_DELTA, START_X, START_Y,
)
from .levels import LEVELS


class GameLogic:
    """Core game state and rules. No rendering — safe to run headless."""

    def __init__(self, level: int = 1):
        self.level_id = level
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._init_grid()
        self._init_snake()
        self._place_food()

        self.bomb: bool = False
        self.bomb_pos: tuple[int, int] | None = None
        self.bomb_timer: int = 0

        self.explosion: bool = False
        self.explosion_pos: tuple[int, int] | None = None
        self.explosion_timer: int = 0

        self.food_timer: int = 0
        self.game_over: bool = False
        self.score: int = 0

    def step(self, action: int) -> tuple[bool, bool]:
        """Einen Spielschritt ausführen.

        Rückgabe: (ate_food, died)
        Reihenfolge: move → cut_tail → tick_explosion → tick_food → tick_bomb
        """
        if self.game_over:
            return False, True

        ate_food, died = self._move_snake(action)

        if not died:
            self._cut_tail_at_explosion()
            self._tick_explosion()          # erst Explosion abbauen …
            if not ate_food:
                self._tick_food()           # … dann Essen prüfen …
            self._tick_bomb()               # … dann Bombe prüfen (kann neue Explosion starten)

        return ate_food, died or self.game_over

    # ------------------------------------------------------------------
    # Initialisierung
    # ------------------------------------------------------------------

    def _init_grid(self) -> None:
        layout = LEVELS[self.level_id]
        self.grid: list[list[FieldType]] = [row[:] for row in layout]

    def _init_snake(self) -> None:
        self.direction: Direction = Direction.RIGHT
        # Kopf bei (START_X, START_Y), Körper erstreckt sich nach links
        self.snake: list[tuple[int, int]] = [
            (START_X,     START_Y),
            (START_X - 1, START_Y),
            (START_X - 2, START_Y),
        ]

    def _place_food(self) -> None:
        """Essen auf einer zufälligen freien Zelle platzieren."""
        while True:
            x = random.randrange(FIELDSIZE)
            y = random.randrange(FIELDSIZE)
            if (self.grid[y][x] == FieldType.FREE
                    and (x, y) not in self.snake):
                self.food_pos: tuple[int, int] = (x, y)
                return

    # ------------------------------------------------------------------
    # Bewegung
    # ------------------------------------------------------------------

    def _move_snake(self, action: int) -> tuple[bool, bool]:
        """Kopf bewegen, Kollisionen prüfen, Essen aufnehmen."""
        new_dir = Direction(action)
        if new_dir != OPPOSITE[self.direction]:
            self.direction = new_dir

        dx, dy = DIR_DELTA[self.direction]
        hx, hy = self.snake[0]
        # % FIELDSIZE = wrap-around: snake exits one side and enters the other
        nx = (hx + dx) % FIELDSIZE
        ny = (hy + dy) % FIELDSIZE

        # Wand- oder Explosionskollision
        if self.grid[ny][nx] != FieldType.FREE:
            self.game_over = True
            return False, True

        # Selbstkollision (Kopf trifft Körper, nicht sich selbst)
        if len(self.snake) > 1 and (nx, ny) in self.snake[1:]:
            self.game_over = True
            return False, True

        ate_food = (nx, ny) == self.food_pos

        if ate_food:
            # Wachsen: neuer Kopf vorne, Schwanz bleibt
            self.snake = [(nx, ny)] + self.snake
            self._place_food()
            self.food_timer = 0
            self.score += 1
        else:
            # Normale Bewegung: neuer Kopf, letztes Segment fällt weg
            self.snake = [(nx, ny)] + self.snake[:-1]

        return ate_food, False

    # ------------------------------------------------------------------
    # Schwanz abschneiden
    # ------------------------------------------------------------------

    def _cut_tail_at_explosion(self) -> None:
        """Alle Schlangensegmente ab dem ersten EXPLODED-Feld entfernen."""
        for i, (x, y) in enumerate(self.snake):
            if self.grid[y][x] == FieldType.EXPLODED:
                if i == 0:  # head is in explosion → instant death
                    self.game_over = True
                    self.snake = []
                else:        # body segment hit → trim tail from that point
                    self.snake = self.snake[:i]
                return

    # ------------------------------------------------------------------
    # Explosion – Grid-Manipulation
    # ------------------------------------------------------------------

    def _cross_field(self, cx: int, cy: int, field_type: FieldType) -> None:
        """Kreuz-Muster um (cx, cy) setzen, stoppt an Wänden (mit Wrap-around).
        Walls block the blast arm but are not destroyed themselves."""
        self.grid[cy][cx] = field_type

        for i in range(1, EXPLOSION_REACH + 1):
            nx = (cx + i) % FIELDSIZE
            if self.grid[cy][nx] == FieldType.WALL:
                break
            self.grid[cy][nx] = field_type

        for i in range(1, EXPLOSION_REACH + 1):
            nx = (cx - i) % FIELDSIZE
            if self.grid[cy][nx] == FieldType.WALL:
                break
            self.grid[cy][nx] = field_type

        for i in range(1, EXPLOSION_REACH + 1):
            ny = (cy + i) % FIELDSIZE
            if self.grid[ny][cx] == FieldType.WALL:
                break
            self.grid[ny][cx] = field_type

        for i in range(1, EXPLOSION_REACH + 1):
            ny = (cy - i) % FIELDSIZE
            if self.grid[ny][cx] == FieldType.WALL:
                break
            self.grid[ny][cx] = field_type

    def _apply_explosion(self) -> None:
        x, y = self.explosion_pos
        self._cross_field(x, y, FieldType.EXPLODED)

    def _clear_explosion(self) -> None:
        x, y = self.explosion_pos
        self._cross_field(x, y, FieldType.FREE)

    # ------------------------------------------------------------------
    # Timer-Ticks
    # ------------------------------------------------------------------

    def _tick_explosion(self) -> None:
        if not self.explosion:
            return
        self.explosion_timer += 1
        if self.explosion_timer >= EXPLOSION_STEPS:
            self.explosion_timer = 0
            self.explosion = False
            self._clear_explosion()

    def _tick_food(self) -> None:
        self.food_timer += 1
        if self.food_timer >= FOOD_STEPS:
            self.bomb = True
            self.bomb_pos = self.food_pos
            self.bomb_timer = 0
            self._place_food()
            self.food_timer = 0

    def _tick_bomb(self) -> None:
        if not self.bomb:
            return
        self.bomb_timer += 1
        if self.bomb_timer >= BOMB_STEPS:
            self.bomb = False
            self.bomb_timer = 0
            self.explosion = True
            self.explosion_pos = self.bomb_pos
            self.explosion_timer = 0
            self._apply_explosion()

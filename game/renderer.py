import numpy as np
import pygame

from .constants import FIELDSIZE, FieldType, Cell
from .logic import GameLogic

# Pixel pro Zelle
CELL_SIZE = 60
WINDOW_SIZE = FIELDSIZE * CELL_SIZE   # 600 × 600

# Farben (RGB) passend zu den Farben aus scene2D.c
_COLORS: dict[int, tuple[int, int, int]] = {
    Cell.FREE:       ( 80,  80,  80),
    Cell.WALL:       ( 60,  60,  60),
    Cell.EXPLODED:   (200,  80,  20),
    Cell.SNAKE_HEAD: (  0, 200,   0),
    Cell.SNAKE_BODY: (  0, 140,   0),
    Cell.FOOD:       (210,  40,  40),
    Cell.BOMB:       ( 35,  35,  35),
}

# Wandmuster – dunklere Ziegel-Linien über dem Wandblock
_WALL_BASE   = ( 60,  60,  60)
_WALL_BRICK  = ( 90,  90,  90)
_WALL_MORTAR = ( 45,  45,  45)


class Renderer:
    def __init__(self, cell_size: int = CELL_SIZE):
        self.cell_size   = cell_size
        self.window_size = FIELDSIZE * cell_size
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._init_pygame()

    # ------------------------------------------------------------------
    # Öffentliche Methoden
    # ------------------------------------------------------------------

    def draw(self, game: GameLogic) -> None:
        """Spielzustand auf den Bildschirm rendern."""
        obs = self._build_obs(game)
        self._draw_cells(obs)
        pygame.display.flip()
        self._clock.tick(10)

        # Events abarbeiten damit das Fenster nicht einfriert
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def get_rgb_array(self, game: GameLogic) -> np.ndarray:
        """Spielzustand als RGB-Array (H×W×3) zurückgeben."""
        obs = self._build_obs(game)
        self._draw_cells(obs)
        return pygame.surfarray.array3d(self._surface).transpose(1, 0, 2)

    def close(self) -> None:
        if self._surface is not None:
            pygame.quit()
            self._surface = None
            self._clock   = None

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        self._surface = pygame.display.set_mode(
            (self.window_size, self.window_size)
        )
        pygame.display.set_caption("Bomberman Snake")
        self._clock = pygame.time.Clock()

    def _build_obs(self, game: GameLogic) -> np.ndarray:
        """Identische Logik wie BombermanSnakeEnv._get_obs()."""
        grid = np.zeros((FIELDSIZE, FIELDSIZE), dtype=np.int8)

        for y in range(FIELDSIZE):
            for x in range(FIELDSIZE):
                ft = game.grid[y][x]
                if ft == FieldType.WALL:
                    grid[y][x] = Cell.WALL
                elif ft == FieldType.EXPLODED:
                    grid[y][x] = Cell.EXPLODED

        if game.bomb and game.bomb_pos is not None:
            bx, by = game.bomb_pos
            grid[by][bx] = Cell.BOMB

        fx, fy = game.food_pos
        if grid[fy][fx] == Cell.FREE:
            grid[fy][fx] = Cell.FOOD

        for x, y in game.snake[1:]:
            grid[y][x] = Cell.SNAKE_BODY

        if game.snake:
            hx, hy = game.snake[0]
            grid[hy][hx] = Cell.SNAKE_HEAD

        return grid

    def _draw_cells(self, obs: np.ndarray) -> None:
        cs = self.cell_size
        self._surface.fill((20, 20, 20))   # Hintergrund

        for y in range(FIELDSIZE):
            # game y=0 ist unten → Pygame-Reihe = FIELDSIZE-1-y
            py = (FIELDSIZE - 1 - y) * cs

            for x in range(FIELDSIZE):
                cell = int(obs[y][x])
                px   = x * cs
                rect = pygame.Rect(px, py, cs, cs)

                if cell == Cell.WALL:
                    self._draw_wall(rect)
                elif cell == Cell.EXPLODED:
                    self._draw_exploded(rect)
                elif cell == Cell.FOOD:
                    self._draw_food(rect)
                elif cell == Cell.BOMB:
                    self._draw_bomb(rect)
                elif cell == Cell.SNAKE_HEAD:
                    self._draw_snake_head(rect)
                elif cell == Cell.SNAKE_BODY:
                    self._draw_snake_body(rect)
                else:
                    # FREE
                    pygame.draw.rect(self._surface, _COLORS[Cell.FREE], rect)

    # ------------------------------------------------------------------
    # Zell-Zeichenroutinen
    # ------------------------------------------------------------------

    def _draw_wall(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, _WALL_BASE, rect)
        cs = self.cell_size
        # Vier Ziegelreihen als horizontale Streifen
        for row in range(4):
            y = rect.top + row * cs // 4
            h = cs // 4 - 1
            offset = (cs // 4) if row % 2 == 1 else 0
            for col in range(-1, 3):
                bx = rect.left + col * cs // 2 + offset + 2
                br = pygame.Rect(bx, y + 1, cs // 2 - 4, h - 2)
                clipped = br.clip(rect)
                if clipped.width > 0 and clipped.height > 0:
                    pygame.draw.rect(self._surface, _WALL_BRICK, clipped)

    def _draw_exploded(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, _COLORS[Cell.FREE], rect)
        cx = rect.centerx
        cy = rect.centery
        r  = self.cell_size // 8
        # Orange-rote Explosionspunkte (wie in scene2D.c)
        for color, dx, dy in [
            ((200,  80,  20),  0,      0),
            ((230, 140,   0), -r * 2,  0),
            ((230, 140,   0),  r * 2,  0),
            ((210,  60,  10),  r,     -r * 2),
            ((210,  60,  10), -r,      r * 2),
            (( 50,  50,  50), -r * 2, -r * 2),
            (( 50,  50,  50),  r * 2,  r * 2),
        ]:
            pygame.draw.circle(self._surface, color, (cx + dx, cy + dy), r)

    def _draw_food(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, _COLORS[Cell.FREE], rect)
        cx  = rect.centerx
        cy  = rect.centery
        r   = self.cell_size // 3
        # Rotes Dreieck (wie drawFood in scene2D.c)
        pts = [
            (cx,     cy - r),
            (cx - r, cy + r // 2),
            (cx + r, cy + r // 2),
        ]
        pygame.draw.polygon(self._surface, _COLORS[Cell.FOOD], pts)

    def _draw_bomb(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self._surface, _COLORS[Cell.FREE], rect)
        cx = rect.centerx
        cy = rect.centery
        r  = self.cell_size // 4
        # Dunkelgrauer Kreis
        pygame.draw.circle(self._surface, _COLORS[Cell.BOMB], (cx, cy), r)
        # Zündschnur (zwei Linien in weiß/rot, wie drawBomb in scene2D.c)
        fuse_tip = (cx + r // 2, cy - r)
        pygame.draw.line(self._surface, (220, 220, 220),
                         (cx, cy - r), fuse_tip, 2)
        pygame.draw.line(self._surface, (200, 30, 30),
                         fuse_tip, (fuse_tip[0] + r // 2, fuse_tip[1] - r // 2), 2)

    def _draw_snake_head(self, rect: pygame.Rect) -> None:
        cx = rect.centerx
        cy = rect.centery
        r  = self.cell_size // 2 - 2
        # Grüner Kreis
        pygame.draw.circle(self._surface, _COLORS[Cell.SNAKE_HEAD], (cx, cy), r)
        # Zwei weiße Augen
        eye_r = max(2, self.cell_size // 12)
        offset = r // 2
        pygame.draw.circle(self._surface, (255, 255, 255),
                           (cx - offset // 2, cy - offset), eye_r)
        pygame.draw.circle(self._surface, (255, 255, 255),
                           (cx + offset // 2, cy - offset), eye_r)

    def _draw_snake_body(self, rect: pygame.Rect) -> None:
        inner = rect.inflate(-6, -6)
        pygame.draw.rect(self._surface, _COLORS[Cell.SNAKE_BODY], inner,
                         border_radius=self.cell_size // 5)

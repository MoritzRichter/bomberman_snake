import numpy as np
import pygame

from .constants import FIELDSIZE, FieldType, Cell, Direction
from .logic import GameLogic

# Pixel pro Zelle
CELL_SIZE = 60
WINDOW_SIZE = FIELDSIZE * CELL_SIZE   # 600 × 600

# Abstand zwischen den Bodenplatten, damit das Grid sichtbar wird
GRID_GAP = 4

# Farben (RGB) passend zu den Farben aus scene2D.c
_COLORS: dict[int, tuple[int, int, int]] = {
    Cell.FREE:       ( 80,  80,  80),
    Cell.WALL:       (150,  40,  40),
    Cell.EXPLODED:   (200,  80,  20),
    Cell.SNAKE_HEAD: (  0, 200,   0),
    Cell.SNAKE_BODY: (  0, 140,   0),
    Cell.FOOD:       (210,  40,  40),
    Cell.BOMB:       ( 35,  35,  35),
}

# Rote Wandfarben – Blöcke, die man nicht betreten kann
_WALL_BASE  = (120,  30,  30)
_WALL_BRICK = (175,  55,  55)

# Hintergrund, der zwischen den Bodenplatten als Grid durchscheint
_BACKGROUND = (20, 20, 20)

# Blickrichtung der Augen in Bildschirm-Koordinaten (game-y zeigt nach oben)
_EYE_DIR = {
    Direction.LEFT:  (-1,  0),
    Direction.RIGHT: ( 1,  0),
    Direction.UP:    ( 0, -1),
    Direction.DOWN:  ( 0,  1),
}


class GridRenderer:
    """Zeichnet das Spielfeld auf eine beliebige Surface — kein eigenes Fenster.

    Wird sowohl vom fensterbasierten `Renderer` (main.py) als auch von play.py
    genutzt, damit beide exakt dasselbe Design zeigen.
    """

    def __init__(self, cell_size: int = CELL_SIZE, background=_BACKGROUND):
        self.cell_size  = cell_size
        self.background = background

    def draw(self, surf: pygame.Surface, game: GameLogic) -> None:
        """Aktuellen Spielzustand auf `surf` zeichnen (obere-linke Ecke = Grid)."""
        obs = self._build_obs(game)
        self._draw_cells(surf, obs, game.direction)

    # ------------------------------------------------------------------
    # Zustand → Zell-Raster
    # ------------------------------------------------------------------

    def _build_obs(self, game: GameLogic) -> np.ndarray:
        """Build a Cell-valued 10×10 grid from the current game state for rendering."""
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

    def _draw_cells(self, surf: pygame.Surface, obs: np.ndarray, direction: Direction) -> None:
        cs = self.cell_size
        # Nur den Grid-Bereich füllen (lässt Platz für ein evtl. Panel daneben)
        grid_px = FIELDSIZE * cs
        surf.fill(self.background, pygame.Rect(0, 0, grid_px, grid_px))

        for y in range(FIELDSIZE):
            # game y=0 ist unten → Pygame-Reihe = FIELDSIZE-1-y
            py = (FIELDSIZE - 1 - y) * cs

            for x in range(FIELDSIZE):
                cell = int(obs[y][x])
                px   = x * cs
                rect = pygame.Rect(px, py, cs, cs)

                if cell == Cell.WALL:
                    # Wände füllen die ganze Zelle → durchgehende Barriere
                    self._draw_wall(surf, rect)
                elif cell == Cell.SNAKE_BODY:
                    # Bodenplatte darunter → Ecken sehen aus wie Boden, kein schwarzer Rand
                    self._draw_floor(surf, rect)
                    self._draw_snake_body(surf, rect)
                elif cell == Cell.SNAKE_HEAD:
                    self._draw_floor(surf, rect)
                    self._draw_snake_head(surf, rect, direction)
                else:
                    # Bodenplatte mit Abstand, dann ggf. Objekt darauf
                    self._draw_floor(surf, rect)
                    if cell == Cell.EXPLODED:
                        self._draw_exploded(surf, rect)
                    elif cell == Cell.FOOD:
                        self._draw_food(surf, rect)
                    elif cell == Cell.BOMB:
                        self._draw_bomb(surf, rect)

    def _draw_floor(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        """Bodenplatte leicht eingerückt zeichnen, damit ein Grid-Spalt bleibt."""
        pygame.draw.rect(surf, _COLORS[Cell.FREE], rect.inflate(-GRID_GAP, -GRID_GAP))

    # ------------------------------------------------------------------
    # Zell-Zeichenroutinen
    # ------------------------------------------------------------------

    def _draw_wall(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surf, _WALL_BASE, rect)
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
                    pygame.draw.rect(surf, _WALL_BRICK, clipped)

    def _draw_exploded(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
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
            pygame.draw.circle(surf, color, (cx + dx, cy + dy), r)

    def _draw_food(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        cx  = rect.centerx
        cy  = rect.centery
        r   = self.cell_size // 3
        # Rotes Dreieck (wie drawFood in scene2D.c)
        pts = [
            (cx,     cy - r),
            (cx - r, cy + r // 2),
            (cx + r, cy + r // 2),
        ]
        pygame.draw.polygon(surf, _COLORS[Cell.FOOD], pts)

    def _draw_bomb(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        cx = rect.centerx
        cy = rect.centery
        r  = self.cell_size // 4
        # Dunkelgrauer Kreis
        pygame.draw.circle(surf, _COLORS[Cell.BOMB], (cx, cy), r)
        # Zündschnur (zwei Linien in weiß/rot, wie drawBomb in scene2D.c)
        fuse_tip = (cx + r // 2, cy - r)
        pygame.draw.line(surf, (220, 220, 220), (cx, cy - r), fuse_tip, 2)
        pygame.draw.line(surf, (200, 30, 30),
                         fuse_tip, (fuse_tip[0] + r // 2, fuse_tip[1] - r // 2), 2)

    def _draw_snake_head(self, surf: pygame.Surface, rect: pygame.Rect, direction: Direction) -> None:
        # Runder Kopf
        cx, cy = rect.center
        r = self.cell_size // 2 - 2
        pygame.draw.circle(surf, _COLORS[Cell.SNAKE_HEAD], (cx, cy), r)

        fx, fy = _EYE_DIR[direction]          # Vorwärtsrichtung (Bildschirm)
        perp_x, perp_y = -fy, fx              # senkrecht dazu (Augen-Abstand)

        eye_r = max(2, self.cell_size // 9)
        fwd   = r * 0.42                      # Augen nach vorne versetzt
        side  = r * 0.42                      # …und seitlich auseinander
        pupil = max(1, eye_r // 2)

        for s in (1, -1):
            ex = cx + fx * fwd + perp_x * side * s
            ey = cy + fy * fwd + perp_y * side * s
            pygame.draw.circle(surf, (255, 255, 255), (int(ex), int(ey)), eye_r)
            # Pupille schaut in Bewegungsrichtung
            pygame.draw.circle(surf, (20, 20, 20),
                               (int(ex + fx * eye_r * 0.5), int(ey + fy * eye_r * 0.5)), pupil)

    def _draw_snake_body(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        # Ganze Zelle füllen, aber mit abgerundeten Ecken (kein schwarzer Rand)
        pygame.draw.rect(surf, _COLORS[Cell.SNAKE_BODY], rect,
                         border_radius=self.cell_size // 4)


class Renderer:
    """Fensterbasierter Renderer für main.py — kapselt Pygame-Fenster + GridRenderer."""

    def __init__(self, cell_size: int = CELL_SIZE):
        self.cell_size   = cell_size
        self.window_size = FIELDSIZE * cell_size
        self._surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._grid = GridRenderer(cell_size)
        self._init_pygame()

    # ------------------------------------------------------------------
    # Öffentliche Methoden
    # ------------------------------------------------------------------

    def draw(self, game: GameLogic) -> None:
        """Spielzustand auf den Bildschirm rendern."""
        self._grid.draw(self._surface, game)
        pygame.display.flip()
        self._clock.tick(10)

        # Drain the event queue so the OS doesn't mark the window as unresponsive.
        # main.py processes events before calling draw(), so the queue is usually empty here.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def get_rgb_array(self, game: GameLogic) -> np.ndarray:
        """Spielzustand als RGB-Array (H×W×3) zurückgeben."""
        self._grid.draw(self._surface, game)
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

"""
play.py — Watch a saved veteran network play Bomberman Snake.

Controls (selection screen):
  Click         — select file / level / options
  Enter         — start game
  1 / 2 / 3    — quick level select
  Q             — quit

Options (buttons on the selection screen):
  Essen: Bombe / bleibt — food turns into a bomb, or just stays until eaten
  Blast: + / X          — explosion spreads in a + or diagonal X shape
  Ansicht: voll / minimal — full main.py design, or simple flat rendering

Controls (game screen):
  R             — restart same veteran & level
  ESC           — back to selection
  Q             — quit
  ↑ / ↓         — increase / decrease game speed
"""
import sys
import os
import pickle
import glob
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evolution'))

from game.logic import GameLogic
from game.constants import FieldType, Cell, FIELDSIZE
from game.renderer import GridRenderer
from network import activate_network
from sensors import MoveHelper, TURN_LEFT, TURN_RIGHT
from profiles import get_profile

# ── Constants ─────────────────────────────────────────────────────────────────

SPEED_STEPS  = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
DEFAULT_TPS  = 5
CELL_PX      = 60
PANEL_W      = 230
WIN_W        = FIELDSIZE * CELL_PX + PANEL_W
WIN_H        = FIELDSIZE * CELL_PX
VETERANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "veterans")

# ── Colours ───────────────────────────────────────────────────────────────────

BG     = (18,  18,  18)
FG     = (210, 210, 210)
DIM    = (100, 100, 100)
ACCENT = (0,   200, 100)
HILITE = (35,  80,  170)
BORDER = (55,  55,  55)

CELL_COLORS = {
    Cell.FREE       : (55,  55,  55),
    Cell.WALL       : (22,  22,  22),
    Cell.EXPLODED   : (200, 75,  15),
    Cell.SNAKE_HEAD : (0,   220, 80),
    Cell.SNAKE_BODY : (0,   155, 55),
    Cell.FOOD       : (220, 45,  45),
    Cell.BOMB       : (170, 110, 0),
}

# ── File loading ──────────────────────────────────────────────────────────────

def load_veteran(path: str):
    """Returns (network, profile_dict, profile_name)."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        network      = data["network"]
        profile_name = data.get("profile_name", "full")
    else:
        # Legacy format: bare network object
        network      = data
        size_map     = {3: "minimal", 6: "basic", 9: "bomb_aware", 12: "timer", 14: "full"}
        profile_name = size_map.get(getattr(network, "input_size", 14), "full")

    return network, get_profile(profile_name), profile_name


def list_veterans() -> list[str]:
    """All .pkl files in veterans/, sorted newest first."""
    return sorted(
        glob.glob(os.path.join(VETERANS_DIR, "*.pkl")),
        key=os.path.getmtime,
        reverse=True,
    )

# ── Grid rendering ────────────────────────────────────────────────────────────

def build_obs(game):
    grid = [[Cell.FREE] * FIELDSIZE for _ in range(FIELDSIZE)]
    for y in range(FIELDSIZE):
        for x in range(FIELDSIZE):
            ft = game.grid[y][x]
            if ft == FieldType.WALL:
                grid[y][x] = Cell.WALL
            elif ft == FieldType.EXPLODED:
                grid[y][x] = Cell.EXPLODED

    if game.bomb and game.bomb_pos:
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


def draw_grid(surface, game):
    obs = build_obs(game)
    for y in range(FIELDSIZE):
        py = (FIELDSIZE - 1 - y) * CELL_PX
        for x in range(FIELDSIZE):
            color = CELL_COLORS.get(obs[y][x], CELL_COLORS[Cell.FREE])
            pygame.draw.rect(surface, color, (x * CELL_PX, py, CELL_PX, CELL_PX))
            pygame.draw.rect(surface, (30, 30, 30), (x * CELL_PX, py, CELL_PX, CELL_PX), 1)


def draw_panel(surface, font, big_font, food_eaten, turns, level, profile_name, file_name,
               longevity, game_over, game_tps=DEFAULT_TPS,
               food_explodes=True, explosion_shape="plus"):
    px = FIELDSIZE * CELL_PX
    pygame.draw.rect(surface, (26, 26, 26), (px, 0, PANEL_W, WIN_H))
    pygame.draw.line(surface, BORDER, (px, 0), (px, WIN_H), 2)

    y = [18]

    def row(text, f=None, color=FG):
        f = f or font
        surface.blit(f.render(text, True, color), (px + 14, y[0]))
        y[0] += f.size(text)[1] + 5

    def sep():
        y[0] += 6
        pygame.draw.line(surface, BORDER, (px + 10, y[0]), (px + PANEL_W - 10, y[0]))
        y[0] += 8

    row("Bomberman Snake", big_font, ACCENT)
    sep()
    row(f"Level:    {level}")
    row(f"Profile:  {profile_name}")
    row(f"Longevity: {longevity} gens", color=DIM)
    row(f"Essen: {'Bombe' if food_explodes else 'bleibt'}   Blast: {explosion_shape.upper()}",
        color=DIM)
    sep()
    row(f"Food eaten:  {food_eaten:5d}")
    row(f"Turns:       {turns:5d}")
    sep()

    speed_str = f"{game_tps:.1f}" if game_tps < 1 else f"{int(game_tps)}"
    row(f"Speed:  {speed_str} steps/s")
    row("↑/↓ — change speed", color=DIM)
    sep()

    # File name (wrapped at ~24 chars)
    row("File:", color=DIM)
    for chunk in [file_name[i:i+22] for i in range(0, len(file_name), 22)]:
        row(chunk, color=DIM)

    if game_over:
        y[0] = WIN_H - 110
        row("GAME OVER", big_font, (220, 55, 55))
        y[0] += 6
        row("R  — restart", color=FG)
        row("ESC — select", color=FG)
        row("Q  — quit",    color=FG)
    else:
        y[0] = WIN_H - 60
        row("R   — restart", color=DIM)
        row("ESC — back",    color=DIM)


# ── Selection screen ──────────────────────────────────────────────────────────

def selection_screen(surface, font, big_font, clock):
    """Returns (path, level, food_explodes, explosion_shape) or None to quit."""
    sel_file        = None
    sel_level       = 1
    food_explodes   = True      # False → Essen bleibt liegen (keine Bombe)
    explosion_shape = "plus"    # "x"   → Explosion in X-Form
    minimal_view    = False     # True  → einfache flache Darstellung
    scroll          = 0

    LIST_TOP    = 190
    ITEM_H      = 30
    LIST_BOTTOM = WIN_H - 70
    MAX_VIS     = (LIST_BOTTOM - LIST_TOP) // ITEM_H

    # Toggle-Buttons für die Spieloptionen (analog zur Level-Auswahl)
    food_btn  = pygame.Rect(90,  118, 150, 36)
    blast_btn = pygame.Rect(250, 118, 150, 36)
    view_btn  = pygame.Rect(410, 118, 150, 36)

    while True:
        files = list_veterans()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return None
                elif event.key == pygame.K_RETURN and sel_file:
                    return sel_file, sel_level, food_explodes, explosion_shape, minimal_view
                elif event.key == pygame.K_UP:
                    scroll = max(0, scroll - 1)
                elif event.key == pygame.K_DOWN:
                    scroll = min(max(0, len(files) - MAX_VIS), scroll + 1)
                elif event.unicode in ("1", "2", "3"):
                    sel_level = int(event.unicode)
            elif event.type == pygame.MOUSEWHEEL:
                scroll = max(0, min(max(0, len(files) - MAX_VIS), scroll - event.y))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Level buttons
                for lvl in (1, 2, 3):
                    bx = 90 + (lvl - 1) * 100
                    if bx <= mx <= bx + 80 and 72 <= my <= 108:
                        sel_level = lvl
                # Option toggle buttons
                if food_btn.collidepoint(mx, my):
                    food_explodes = not food_explodes
                if blast_btn.collidepoint(mx, my):
                    explosion_shape = "x" if explosion_shape == "plus" else "plus"
                if view_btn.collidepoint(mx, my):
                    minimal_view = not minimal_view
                # File list
                for i in range(MAX_VIS):
                    idx = scroll + i
                    if idx >= len(files):
                        break
                    fy = LIST_TOP + i * ITEM_H
                    if fy <= my <= fy + ITEM_H:
                        sel_file = files[idx]
                # Play button
                play_r = pygame.Rect(WIN_W // 2 - 80, WIN_H - 56, 160, 42)
                if play_r.collidepoint(mx, my) and sel_file:
                    return sel_file, sel_level, food_explodes, explosion_shape, minimal_view

        # ── Draw ──────────────────────────────────────────────────────────
        surface.fill(BG)

        surface.blit(big_font.render("Bomberman Snake — Watch Mode", True, ACCENT), (20, 20))

        # Level buttons
        surface.blit(font.render("Level:", True, FG), (20, 82))
        for lvl in (1, 2, 3):
            bx  = 90 + (lvl - 1) * 100
            col = HILITE if sel_level == lvl else (45, 45, 45)
            pygame.draw.rect(surface, col, (bx, 72, 80, 36), border_radius=6)
            pygame.draw.rect(surface, BORDER, (bx, 72, 80, 36), 1, border_radius=6)
            lbl = font.render(f"Level {lvl}", True, FG)
            surface.blit(lbl, lbl.get_rect(center=(bx + 40, 90)))

        # Option toggle buttons (highlighted when in the non-default mode)
        surface.blit(font.render("Optionen:", True, FG), (20, 128))
        for rect, active, text in (
            (food_btn,  not food_explodes,        "Essen: bleibt" if not food_explodes else "Essen: Bombe"),
            (blast_btn, explosion_shape == "x",   "Blast: X" if explosion_shape == "x" else "Blast: +"),
            (view_btn,  minimal_view,             "Ansicht: minimal" if minimal_view else "Ansicht: voll"),
        ):
            col = HILITE if active else (45, 45, 45)
            pygame.draw.rect(surface, col, rect, border_radius=6)
            pygame.draw.rect(surface, BORDER, rect, 1, border_radius=6)
            lbl = font.render(text, True, FG)
            surface.blit(lbl, lbl.get_rect(center=rect.center))

        # List header
        surface.blit(font.render("Veterans  (newest first — click to select):", True, DIM), (20, LIST_TOP - 24))

        if not files:
            msg = f"No .pkl files found in  {VETERANS_DIR}/"
            surface.blit(font.render(msg, True, (160, 70, 70)), (20, LIST_TOP + 10))
        else:
            for i in range(MAX_VIS):
                idx = scroll + i
                if idx >= len(files):
                    break
                fy   = LIST_TOP + i * ITEM_H
                path = files[idx]
                name = os.path.basename(path)
                is_s = path == sel_file
                if is_s:
                    pygame.draw.rect(surface, HILITE, (10, fy, WIN_W - 20, ITEM_H - 2), border_radius=4)
                surface.blit(font.render(name, True, FG if is_s else DIM), (22, fy + 6))

        # Scroll hint
        if len(files) > MAX_VIS:
            surface.blit(font.render(f"↑↓ scroll  ({scroll+1}–{min(scroll+MAX_VIS,len(files))} of {len(files)})", True, DIM),
                         (20, LIST_BOTTOM + 6))

        # Play button
        play_r = pygame.Rect(WIN_W // 2 - 80, WIN_H - 56, 160, 42)
        active = sel_file is not None
        pygame.draw.rect(surface, ACCENT if active else (45, 45, 45), play_r, border_radius=8)
        pygame.draw.rect(surface, BORDER, play_r, 1, border_radius=8)
        lbl = font.render("▶  Play", True, BG if active else DIM)
        surface.blit(lbl, lbl.get_rect(center=play_r.center))

        pygame.display.flip()
        clock.tick(30)


# ── Game runner ───────────────────────────────────────────────────────────────

def _do_tick(game, network, profile):
    """Run one game step; returns (ate, game_over)."""
    helper  = MoveHelper(game, profile)
    inputs  = helper.get_inputs()
    outputs = activate_network(network, inputs)

    turn_left  = round(outputs[0])
    turn_right = round(outputs[1])

    if turn_left:
        direction = TURN_LEFT[game.direction]
    elif turn_right:
        direction = TURN_RIGHT[game.direction]
    else:
        direction = game.direction

    ate, died = game.step(int(direction))
    return ate, died or game.game_over


def run_game(surface, font, big_font, clock, veteran_path, level,
             food_explodes=True, explosion_shape="plus", minimal_view=False):
    network, profile, profile_name = load_veteran(veteran_path)
    longevity  = getattr(network, "longevity", "?")
    file_name  = os.path.basename(veteran_path)
    speed_idx  = SPEED_STEPS.index(DEFAULT_TPS)
    game_tps   = SPEED_STEPS[speed_idx]
    grid_rend  = GridRenderer(CELL_PX)   # Voll-Design, teilt Code mit main.py

    restart = True
    while restart:
        restart    = False
        game       = GameLogic(level, food_explodes=food_explodes,
                               explosion_shape=explosion_shape)
        food_eaten = 0
        turns      = 0
        game_over  = False
        last_tick  = pygame.time.get_ticks() - 10000  # fire immediately
        running    = True

        while running:
            # ── Events ────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "quit"
                    elif event.key == pygame.K_ESCAPE:
                        return "select"
                    elif event.key == pygame.K_r:
                        restart = True
                        running = False
                    elif event.key == pygame.K_UP:
                        speed_idx = min(len(SPEED_STEPS) - 1, speed_idx + 1)
                        game_tps  = SPEED_STEPS[speed_idx]
                    elif event.key == pygame.K_DOWN:
                        speed_idx = max(0, speed_idx - 1)
                        game_tps  = SPEED_STEPS[speed_idx]

            if not running:
                break

            # ── Game tick(s) ──────────────────────────────────────────────
            now = pygame.time.get_ticks()
            if not game_over:
                if game_tps <= 60:
                    # Slow / moderate: at most one tick per frame
                    ms_per_tick = 1000.0 / game_tps
                    if now - last_tick >= ms_per_tick:
                        last_tick = now
                        ate, over = _do_tick(game, network, profile)
                        turns += 1
                        if ate:
                            food_eaten += 1
                        if over:
                            game_over = True
                else:
                    # Fast: run as many ticks as fit in ~14 ms frame budget
                    deadline = now + 14
                    while pygame.time.get_ticks() < deadline and not game_over:
                        ate, over = _do_tick(game, network, profile)
                        turns += 1
                        if ate:
                            food_eaten += 1
                        if over:
                            game_over = True

            # ── Render ────────────────────────────────────────────────────
            surface.fill(BG)
            if minimal_view:
                draw_grid(surface, game)          # einfache flache Darstellung
            else:
                grid_rend.draw(surface, game)     # volles Design wie in main.py
            draw_panel(surface, font, big_font, food_eaten, turns, level,
                       profile_name, file_name, longevity, game_over, game_tps,
                       food_explodes, explosion_shape)
            pygame.display.flip()
            clock.tick(60)

    return "select"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    pygame.init()
    surface  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Bomberman Snake — Watch Mode")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont(None, 22)
    big_font = pygame.font.SysFont(None, 28)

    while True:
        result = selection_screen(surface, font, big_font, clock)
        if result is None:
            break
        path, level, food_explodes, explosion_shape, minimal_view = result
        action = run_game(surface, font, big_font, clock, path, level,
                          food_explodes, explosion_shape, minimal_view)
        if action == "quit":
            break

    pygame.quit()


if __name__ == "__main__":
    main()

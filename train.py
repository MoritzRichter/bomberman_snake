import sys
import os
import csv
import math
import pickle
import statistics
from datetime import datetime
import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evolution'))

from game.constants import FieldType, Cell, FIELDSIZE
from network import buildNetwork
from evolution import sortPopulation, getOffspring, mutatePopulation, createPopulation
from constants import config
from agent import Agent

# ── Training config ───────────────────────────────────────────────────────

GAMES_COUNT     = 50
GENERATIONS     = 100
LEVEL           = 1
TICKS_PER_FRAME = 5
ELITISM         = round(0.2 * GAMES_COUNT)
SAVE_PATH       = "best_network.pkl"

# ── Layout ────────────────────────────────────────────────────────────────

GAMES_PER_ROW = 10
CELL_PX       = 8
GAME_PX       = FIELDSIZE * CELL_PX          # 80 px
GAP           = 4
HEADER_H      = 30
ROWS          = math.ceil(GAMES_COUNT / GAMES_PER_ROW)

WIN_W = GAMES_PER_ROW * (GAME_PX + GAP) + GAP
WIN_H = ROWS          * (GAME_PX + GAP) + GAP + HEADER_H

# ── Colours ───────────────────────────────────────────────────────────────

BG  = (20,  20,  20)
FG  = (200, 200, 200)
DIM = (100, 100, 100)

CELL_COLORS = {
    Cell.FREE       : ( 80,  80,  80),
    Cell.WALL       : ( 40,  40,  40),
    Cell.EXPLODED   : (200,  80,  20),
    Cell.SNAKE_HEAD : (  0, 200,   0),
    Cell.SNAKE_BODY : (  0, 140,   0),
    Cell.FOOD       : (210,  40,  40),
    Cell.BOMB       : ( 35,  35,  35),
}


# ── Observation builder ───────────────────────────────────────────────────

def build_obs(game):
    """Build a Cell-valued 10×10 array from GameLogic state (mirrors Renderer._build_obs)."""
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


# ── Mini game renderer ────────────────────────────────────────────────────

def draw_game(surface, game, ox, oy, dimmed=False):
    obs = build_obs(game)
    for y in range(FIELDSIZE):
        py = oy + (FIELDSIZE - 1 - y) * CELL_PX    # flip y-axis (game y=0 is bottom row)
        for x in range(FIELDSIZE):
            color = CELL_COLORS.get(int(obs[y][x]), CELL_COLORS[Cell.FREE])
            if dimmed:
                color = tuple(c // 3 for c in color)
            pygame.draw.rect(surface, color, (ox + x * CELL_PX, py, CELL_PX, CELL_PX))


# ── Simulation view ───────────────────────────────────────────────────────

def draw_simulation(surface, font, agents, generation):
    surface.fill(BG)

    alive = sum(1 for a in agents if not a.done)
    header = font.render(
        f"Generation {generation + 1} / {GENERATIONS}   alive: {alive} / {GAMES_COUNT}",
        True, FG,
    )
    surface.blit(header, (8, 8))

    for i, agent in enumerate(agents):
        row = i // GAMES_PER_ROW
        col = i % GAMES_PER_ROW
        ox  = GAP + col * (GAME_PX + GAP)
        oy  = HEADER_H + GAP + row * (GAME_PX + GAP)
        draw_game(surface, agent.game, ox, oy, dimmed=agent.done)

    pygame.display.flip()


# ── Results screen ────────────────────────────────────────────────────────

def draw_results(surface, font, big_font, results, best_ever):
    LINE_H   = 18
    LIST_TOP = 58
    max_scroll = max(0, len(results) * LINE_H - (WIN_H - LIST_TOP))
    scroll_y   = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEWHEEL:
                scroll_y = max(0, min(max_scroll, scroll_y - event.y * LINE_H * 3))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    scroll_y = max(0, scroll_y - LINE_H)
                elif event.key == pygame.K_DOWN:
                    scroll_y = min(max_scroll, scroll_y + LINE_H)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()

        surface.fill(BG)

        title = big_font.render(f"Training complete  —  best score ever: {best_ever:.2f}", True, FG)
        surface.blit(title, (8, 8))

        sub = font.render("Scroll / ↑↓ to navigate   Q to quit", True, DIM)
        surface.blit(sub, (8, 36))

        for i, row in enumerate(results):
            y = LIST_TOP + i * LINE_H - scroll_y
            if LIST_TOP <= y < WIN_H:
                line = font.render(
                    f"Gen {row['generation']:3d}   "
                    f"best: {row['best_score']:8.2f}   "
                    f"mean: {row['mean_score']:7.2f}   "
                    f"median: {row['median_score']:7.2f}   "
                    f"worst: {row['worst_score']:8.2f}",
                    True, FG,
                )
                surface.blit(line, (8, y))

        pygame.display.flip()


# ── Persistence ───────────────────────────────────────────────────────────

def save_best(network):
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(network, f)
    print(f"Best network saved → {SAVE_PATH}")


def write_csv(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = f"training_report_{timestamp}.csv"

    fields = ["generation", "best_score", "mean_score", "median_score", "worst_score"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"Training report saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    surface  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Bomberman Snake — Evolution")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont(None, 16)
    big_font = pygame.font.SysFont(None, 24)

    brains = createPopulation(buildNetwork(9, 2), GAMES_COUNT)

    agents = [
        Agent(
            level                = LEVEL,
            max_turns            = config.max_turns,
            lowest_score_allowed = config.lowest_score_allowed,
            on_game_over         = lambda: None,
        )
        for _ in range(GAMES_COUNT)
    ]

    results         = []
    best_score_ever = float("-inf")
    best_network    = None

    for generation in range(GENERATIONS):

        # ── initialise all agents ────────────────────────────────────────
        for i, agent in enumerate(agents):
            agent.start(brains[i])

        # ── tick until every agent is done ───────────────────────────────
        while not all(agent.done for agent in agents):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            for _ in range(TICKS_PER_FRAME):
                for agent in agents:
                    agent.tick()

            draw_simulation(surface, font, agents, generation)
            clock.tick(60)

        draw_simulation(surface, font, agents, generation)   # final frame

        # ── stats ────────────────────────────────────────────────────────
        sorted_brains = sortPopulation(brains)
        scores        = [b.score for b in brains]
        best_score    = sorted_brains[0].score
        worst_score   = sorted_brains[-1].score
        mean_score    = statistics.mean(scores)
        median_score  = statistics.median(scores)

        results.append({
            "generation"   : generation + 1,
            "best_score"   : round(best_score,   4),
            "mean_score"   : round(mean_score,   4),
            "median_score" : round(median_score, 4),
            "worst_score"  : round(worst_score,  4),
        })

        print(
            f"Gen {generation + 1:3d}  |"
            f"  best: {best_score:8.2f}  |"
            f"  mean: {mean_score:7.2f}  |"
            f"  median: {median_score:7.2f}  |"
            f"  worst: {worst_score:8.2f}"
        )

        if best_score > best_score_ever:
            best_score_ever = best_score
            best_network    = sorted_brains[0]

        # ── evolve (skip on final generation) ────────────────────────────
        if generation < GENERATIONS - 1:
            elitists  = sorted_brains[:ELITISM]
            offspring = [getOffspring(sorted_brains) for _ in range(GAMES_COUNT - ELITISM)]
            mutated   = mutatePopulation(offspring, config.mutation_rate, config.mutation_amount)
            brains    = elitists + mutated

    save_best(best_network)
    write_csv(results)
    draw_results(surface, font, big_font, results, best_score_ever)


if __name__ == "__main__":
    main()

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
from constants import config, apply_scoring_preset
from agent import Agent
from profiles import get_profile, profile_input_size

# ── Training config ───────────────────────────────────────────────────────

GAMES_COUNT         = 50
GENERATIONS         = 600
LEVEL               = 1
RENDER_FPS          = 60       # display refresh rate; game logic runs uncapped
PROFILE_NAME        = "raw"   # "basic" | "bomb_aware" | "timer" | "full" | "raw"
SCORING_MODE        = "balanced"  # "balanced" | "survival" | "food" | "length" | "in_game_score"
SELECTION_STRATEGY  = "power"  # "power" | "tournament" | "roulette" | "top_n" | "random"
ELITISM_RATE        = 0.2      # fraction of population that survives unchanged (0.0–0.5)
_HERE               = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH           = os.path.join(_HERE, "models", "best_network.pkl")
VETERANS_DIR        = os.path.join(_HERE, "models", "veterans")
SEEDS_DIR           = os.path.join(_HERE, "models", "seeds")
TRAINING_REPORT_DIR = os.path.join(_HERE, "runs", "reports")
SEED_PATH    = None   # set to a models/seeds/*.pkl path to warm-start from a previous run

ELITISM = round(ELITISM_RATE * GAMES_COUNT)

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

def draw_simulation(surface, font, agents, generation, steps_per_sec=0):
    surface.fill(BG)

    alive = sum(1 for a in agents if not a.done)
    header = font.render(
        f"Generation {generation + 1} / {GENERATIONS}   alive: {alive} / {GAMES_COUNT}"
        f"   {steps_per_sec:,} steps/s",
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
    max_scroll = max(0, len(results) * LINE_H * 2 - (WIN_H - LIST_TOP))
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
            y = LIST_TOP + i * LINE_H * 2 - scroll_y
            if LIST_TOP <= y < WIN_H:
                score_line = font.render(
                    f"Gen {row['generation']:3d}  "
                    f"score  best: {row['best_score']:8.2f}  "
                    f"mean: {row['mean_score']:7.2f}  "
                    f"median: {row['median_score']:7.2f}  "
                    f"worst: {row['worst_score']:8.2f}",
                    True, FG,
                )
                turns_line = font.render(
                    f"           "
                    f"turns  best: {row['best_turns']:5d}        "
                    f"         median: {row['median_turns']:7.1f}  "
                    f"worst: {row['worst_turns']:5d}",
                    True, DIM,
                )
                surface.blit(score_line, (8, y))
                surface.blit(turns_line, (8, y + LINE_H))

        pygame.display.flip()


# ── Persistence ───────────────────────────────────────────────────────────

def save_best(network):
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(network, f)
    print(f"Best network saved → {SAVE_PATH}")


def save_veteran(network, profile_name: str):
    if network is None:
        return
    os.makedirs(VETERANS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VETERANS_DIR, f"veteran_{timestamp}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"network": network, "profile_name": profile_name, "longevity": network.longevity}, f)
    print(f"Veteran network saved → {path}  (elite for {network.longevity} generations)")


def save_seeds(brains: list, profile_name: str):
    os.makedirs(SEEDS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SEEDS_DIR, f"elite_{timestamp}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"brains": brains, "profile_name": profile_name}, f)
    print(f"Elite seeds saved → {path}  ({len(brains)} brains)")


def load_seeds(path: str, n_inputs: int) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    raw   = data.get("brains", []) if isinstance(data, dict) else data
    valid = [b for b in raw if getattr(b, "input_size", None) == n_inputs]
    if not valid:
        print(f"Seed file has no brains with input_size={n_inputs} — starting fresh")
        return []
    print(f"Loaded {len(valid)} seed brain(s) from {path}")
    return valid


def write_csv(results):
    os.makedirs(TRAINING_REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(TRAINING_REPORT_DIR, f"training_report_{timestamp}.csv")

    fields = [
        "generation",
        "best_score", "mean_score", "median_score", "worst_score",
        "best_turns", "median_turns", "worst_turns",
        "best_food_eaten",
        "best_sc_survival", "best_sc_towards", "best_sc_against",
        "best_sc_ate", "best_sc_bomb",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"Training report saved → {path}")


# ── End-of-training graphs ────────────────────────────────────────────────

def show_graphs(results):
    import matplotlib.pyplot as plt

    gens          = [r["generation"]      for r in results]
    best_scores   = [r["best_score"]      for r in results]
    mean_scores   = [r["mean_score"]      for r in results]
    median_scores = [r["median_score"]    for r in results]
    worst_scores  = [r["worst_score"]     for r in results]
    best_turns    = [r["best_turns"]      for r in results]
    median_turns  = [r["median_turns"]    for r in results]
    worst_turns   = [r["worst_turns"]     for r in results]
    food_eaten    = [r.get("best_food_eaten",  0) for r in results]
    sc_survival   = [r.get("best_sc_survival", 0) for r in results]
    sc_towards    = [r.get("best_sc_towards",  0) for r in results]
    sc_against    = [r.get("best_sc_against",  0) for r in results]
    sc_ate        = [r.get("best_sc_ate",       0) for r in results]
    sc_bomb       = [r.get("best_sc_bomb",      0) for r in results]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Bomberman Snake — Training Overview", fontsize=13, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Score overview (top-left) ─────────────────────────────────────
    ax1.plot(gens, best_scores,   color="#00e676", linewidth=1.5, label="Best")
    ax1.plot(gens, median_scores, color="#ffd740", linewidth=1.5, label="Median")
    ax1.plot(gens, mean_scores,   color="#40c4ff", linewidth=1.0, linestyle="--", label="Mean", alpha=0.8)
    ax1.plot(gens, worst_scores,  color="#ff5252", linewidth=1.0, label="Worst", alpha=0.6)
    ax1.fill_between(gens, best_scores, worst_scores, alpha=0.07, color="white")
    ax1.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax1.set_title("Score (population)", fontsize=10)
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Turns survived (top-right) ────────────────────────────────────
    ax2.plot(gens, best_turns,   color="#00e676", linewidth=1.5, label="Best")
    ax2.plot(gens, median_turns, color="#ffd740", linewidth=1.5, label="Median")
    ax2.plot(gens, worst_turns,  color="#ff5252", linewidth=1.0, label="Worst", alpha=0.6)
    ax2.fill_between(gens, best_turns, worst_turns, alpha=0.07, color="white")
    ax2_r = ax2.twinx()
    ax2_r.bar(gens, food_eaten, color="#ff9800", alpha=0.3, width=0.8, label="Food eaten (best)")
    ax2_r.set_ylabel("Food eaten", color="#ff9800", fontsize=8)
    ax2_r.tick_params(axis="y", labelcolor="#ff9800")
    ax2.set_title("Turns survived + food eaten (best agent)", fontsize=10)
    ax2.set_ylabel("Turns")
    ax2.legend(loc="upper left", fontsize=8)
    ax2_r.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # ── Score breakdown — positive components (bottom-left) ──────────
    ax3.plot(gens, sc_survival, color="#40c4ff", linewidth=1.5, label="Survival")
    ax3.plot(gens, sc_towards,  color="#00e676", linewidth=1.5, label="→ Food (towards)")
    ax3.plot(gens, sc_ate,      color="#ffd740", linewidth=1.5, label="Ate food")
    ax3.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax3.set_title("Score breakdown — gains (best agent)", fontsize=10)
    ax3.set_ylabel("Score contribution")
    ax3.set_xlabel("Generation")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ── Score breakdown — penalties (bottom-right) ────────────────────
    ax4.plot(gens, sc_against, color="#ff5252", linewidth=1.5, label="← Food (away)")
    ax4.plot(gens, sc_bomb,    color="#ff9800", linewidth=1.5, label="Bomb penalty")
    ax4.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax4.set_title("Score breakdown — penalties (best agent)", fontsize=10)
    ax4.set_ylabel("Score contribution")
    ax4.set_xlabel("Generation")
    ax4.legend(loc="lower left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    surface  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Bomberman Snake — Evolution")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont(None, 16)
    big_font = pygame.font.SysFont(None, 24)

    apply_scoring_preset(SCORING_MODE)

    profile  = get_profile(PROFILE_NAME)
    n_inputs = profile_input_size(profile)

    seeds  = load_seeds(SEED_PATH, n_inputs) if SEED_PATH else []
    fresh  = createPopulation(buildNetwork(n_inputs, 2), max(0, GAMES_COUNT - len(seeds)))
    brains = seeds + fresh
    for b in brains:
        b.longevity = getattr(b, "longevity", 0)

    agents = [
        Agent(
            level                = LEVEL,
            max_turns            = config.max_turns,
            lowest_score_allowed = config.lowest_score_allowed,
            on_game_over         = lambda: None,
            profile              = profile,
        )
        for _ in range(GAMES_COUNT)
    ]

    results         = []
    best_score_ever = float("-inf")
    best_network    = None
    veteran_brain   = None

    for generation in range(GENERATIONS):

        # ── initialise all agents ────────────────────────────────────────
        for i, agent in enumerate(agents):
            agent.start(brains[i])

        # ── tick until every agent is done (uncapped, render at RENDER_FPS) ──
        _MS_PER_FRAME = 1000 // RENDER_FPS
        steps_per_sec = 0

        while not all(agent.done for agent in agents):
            frame_start = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Run as many full rounds as possible within one frame budget.
            # Each round ticks every still-alive agent once.
            deadline = frame_start + _MS_PER_FRAME - 2   # 2 ms headroom for rendering
            rounds   = 0
            while pygame.time.get_ticks() < deadline:
                all_done = True
                for agent in agents:
                    if not agent.done:
                        agent.tick()
                        all_done = False
                rounds += 1
                if all_done:
                    break

            steps_per_sec = rounds * RENDER_FPS
            draw_simulation(surface, font, agents, generation, steps_per_sec)
            clock.tick(RENDER_FPS)

        draw_simulation(surface, font, agents, generation, steps_per_sec)   # final frame

        # ── stats ────────────────────────────────────────────────────────
        sorted_brains = sortPopulation(brains)
        scores        = [b.score for b in brains]
        best_score    = sorted_brains[0].score
        worst_score   = sorted_brains[-1].score
        mean_score    = statistics.mean(scores)
        median_score  = statistics.median(scores)

        turns_list    = [a.turns for a in agents]
        best_turns    = max(turns_list)
        worst_turns   = min(turns_list)
        median_turns  = statistics.median(turns_list)

        # Score breakdown for the best-scoring agent
        ba        = next(a for a in agents if a.brain is sorted_brains[0])
        pos_total = ba.score_survival + ba.score_towards + ba.score_ate

        def _pct(v):
            return f"({v / pos_total * 100:.0f}%)" if pos_total > 0 else ""

        results.append({
            "generation"       : generation + 1,
            "best_score"       : round(best_score,        4),
            "mean_score"       : round(mean_score,        4),
            "median_score"     : round(median_score,      4),
            "worst_score"      : round(worst_score,       4),
            "best_turns"       : best_turns,
            "median_turns"     : round(median_turns,      1),
            "worst_turns"      : worst_turns,
            "best_food_eaten"  : ba.food_eaten,
            "best_sc_survival" : round(ba.score_survival, 3),
            "best_sc_towards"  : round(ba.score_towards,  3),
            "best_sc_against"  : round(ba.score_against,  3),
            "best_sc_ate"      : round(ba.score_ate,      3),
            "best_sc_bomb"     : round(ba.score_bomb,     3),
        })

        vet_gens = veteran_brain.longevity if veteran_brain else 0
        print(
            f"Gen {generation + 1:3d}  |"
            f"  score  best: {best_score:8.2f}  mean: {mean_score:7.2f}"
            f"  median: {median_score:7.2f}  worst: {worst_score:8.2f}  |"
            f"  turns  best: {best_turns:5d}  median: {median_turns:7.1f}  worst: {worst_turns:5d}"
            f"  |  veteran: {vet_gens} gens"
        )
        print(
            f"         └ best [food:{ba.food_eaten:3d}  turns:{ba.turns:5d}]:"
            f"  surv {ba.score_survival:+7.2f} {_pct(ba.score_survival):5s}"
            f"  →food {ba.score_towards:+7.2f} {_pct(ba.score_towards):5s}"
            f"  ate {ba.score_ate:+7.2f} {_pct(ba.score_ate):5s}"
            f"  ←food {ba.score_against:+7.2f}"
            f"  bomb {ba.score_bomb:+6.2f}"
        )

        if best_score > best_score_ever:
            best_score_ever = best_score
            best_network    = sorted_brains[0]

        # ── longevity: increment elitists, track veteran ──────────────────
        for b in sorted_brains[:ELITISM]:
            b.longevity += 1
            if veteran_brain is None or b.longevity > veteran_brain.longevity:
                veteran_brain = b

        # ── evolve (skip on final generation) ────────────────────────────
        if generation < GENERATIONS - 1:
            elitists  = sorted_brains[:ELITISM]
            offspring = [getOffspring(sorted_brains, strategy=SELECTION_STRATEGY) for _ in range(GAMES_COUNT - ELITISM)]
            mutated   = mutatePopulation(offspring, config.mutation_rate, config.mutation_amount)
            for b in mutated:
                b.longevity = 0
            brains    = elitists + mutated

    save_best(best_network)
    save_veteran(veteran_brain, PROFILE_NAME)
    save_seeds(sorted_brains[:max(ELITISM, 1)], PROFILE_NAME)
    write_csv(results)
    show_graphs(results)
    draw_results(surface, font, big_font, results, best_score_ever)


if __name__ == "__main__":
    main()

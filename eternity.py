"""
eternity.py — Continuous self-improving training loop (no rendering).

Phase 1 · Bootstrap
    Train 300 generations from scratch.
    Check: max(best_score in last 5 gens) > BOOTSTRAP_THRESHOLD.
    If not → restart from scratch.

Phase 2 · Evolution
    Use elite seeds from the last successful run.
    Train 300 generations.
    Check: median(median_score of last 5 gens) >= prev_run_median × 1.05.
    If yes  → update seed & prev_median, continue.
    If no   → retry with the same seed (failure_count += 1).
    After MAX_FAILURES consecutive failures:
        Save Eternity package to  runs/eternity/<timestamp>/
        Restart from Phase 1.

Outputs per saved package (inside runs/eternity/<timestamp>_run<N>/):
    veteran_<timestamp>.pkl
    elite_<timestamp>.pkl
    training_report_<timestamp>.csv   (from the last improving run)
"""

import sys
import os
import csv
import pickle
import shutil
import statistics
from datetime import datetime
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evolution"))

from game.constants import FieldType, Cell, FIELDSIZE
from network import buildNetwork
from evolution import sortPopulation, getOffspring, mutatePopulation, createPopulation
from constants import config, apply_scoring_preset
from agent import Agent
from profiles import get_profile, profile_input_size

# ── Config ─────────────────────────────────────────────────────────────────────

GAMES_COUNT          = 50
GENERATIONS_PER_RUN  = 300
LEVEL                = 1
PROFILE_NAME         = "raw"

SELECTION_STRATEGY   = "power"      # "power" or "tournament"
SELECTION_POWER      = 4            # bias exponent for power strategy (higher = stronger top-bias)
TOURNAMENT_K         = 5            # candidates drawn per tournament (higher = more selective)
SCORING_MODE         = "balanced"   # "balanced" | "survival" | "food" | "length"
ELITISM_RATE         = 0.2

BOOTSTRAP_THRESHOLD  = 1000.0   # bootstrap passes when best_score > this in last WINDOW gens
IMPROVEMENT_FACTOR   = 1.05     # each evolution run must raise median by this factor
WINDOW               = 5        # tail window (number of gens) for evaluating a run
MAX_FAILURES         = 5        # consecutive failures before saving and restarting

# Multi-game evaluation: each brain plays EVAL_GAMES games; score is the average.
# Reduces luck-based variance in fitness (lucky/unlucky food placement, bomb timing).
EVAL_GAMES = 3

# Level rotation: when True each brain plays one game on each of levels 1, 2, 3
# (overrides LEVEL and sets EVAL_GAMES implicitly to 3).
# Produces more robust agents at the cost of 3× eval time per generation.
LEVEL_ROTATION = False

# Parallel workers: None = all logical CPU cores, 1 = sequential (no overhead)
N_WORKERS = None

# Temp-Elite system: high-performing offspring get protected slots for a few generations
MAX_TEMP_ELITE     = ELITISM_RATE * 0.5     # max extra (non-permanent) slots
TEMP_PROMOTE_AFTER = 2     # how many consecutive good runs before promotion to permanent elite

# Set True to print Temp-Elite promotions and other internal events
VERBOSE = False

VISUAL           = False   # True = show best brain in a live pygame window between generations
RENDER_FPS       = 60      # display refresh rate for the visual window
MAX_VISUAL_TICKS = 3000    # cap each inter-generation demo at this many game ticks

_HERE           = os.path.dirname(os.path.abspath(__file__))
ETERNITY_DIR    = os.path.join(_HERE, "runs", "eternity")
_SUMMARY_PATH   = os.path.join(ETERNITY_DIR, "eternity_summary.csv")
_CHECKPOINT_DIR = os.path.join(ETERNITY_DIR, "checkpoint")
_PROGRESS_PATH  = os.path.join(_CHECKPOINT_DIR, "progress.csv")

ELITISM = round(ELITISM_RATE * GAMES_COUNT)


# ── Visual best-brain demo (optional, pygame) ─────────────────────────────────

_V_CELL_PX  = 24
_V_HEADER_H = 50

_V_BG     = (20, 20, 20)
_V_FG     = (200, 200, 200)
_V_DIM    = (100, 100, 100)
_V_COLORS = {
    Cell.FREE       : ( 80,  80,  80),
    Cell.WALL       : ( 40,  40,  40),
    Cell.EXPLODED   : (200,  80,  20),
    Cell.SNAKE_HEAD : (  0, 200,   0),
    Cell.SNAKE_BODY : (  0, 140,   0),
    Cell.FOOD       : (210,  40,  40),
    Cell.BOMB       : ( 80,  80,  20),
}


def _build_obs(game):
    grid = [[Cell.FREE] * FIELDSIZE for _ in range(FIELDSIZE)]
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


def _run_visual_demo(brain, visual_ctx: dict, header: str):
    """Run one live game of `brain` in the pygame window. Called between training generations.
    The window is frozen while pool.map() is running — this only animates between generations.
    """
    import pygame
    surface = visual_ctx["surface"]
    font    = visual_ctx["font"]
    clock   = visual_ctx["clock"]

    profile = get_profile(PROFILE_NAME)
    apply_scoring_preset(SCORING_MODE)
    agent = Agent(LEVEL, config.max_turns, config.lowest_score_allowed, lambda: None, profile)
    agent.start(brain)

    _MS_PER_FRAME = 1000 // RENDER_FPS
    ticks = 0

    while not agent.done and ticks < MAX_VISUAL_TICKS:
        frame_start = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        deadline = frame_start + _MS_PER_FRAME - 2
        while pygame.time.get_ticks() < deadline and not agent.done and ticks < MAX_VISUAL_TICKS:
            agent.tick()
            ticks += 1

        surface.fill(_V_BG)
        obs = _build_obs(agent.game)
        for y in range(FIELDSIZE):
            py = _V_HEADER_H + (FIELDSIZE - 1 - y) * _V_CELL_PX
            for x in range(FIELDSIZE):
                color = _V_COLORS.get(int(obs[y][x]), _V_COLORS[Cell.FREE])
                pygame.draw.rect(surface, color, (x * _V_CELL_PX, py, _V_CELL_PX, _V_CELL_PX))

        surface.blit(font.render(header, True, _V_FG), (4, 4))
        surface.blit(
            font.render(
                f"food: {agent.food_eaten}   turns: {agent.turns}   score: {agent.brain.score:.1f}",
                True, _V_DIM,
            ),
            (4, 26),
        )
        pygame.display.flip()
        clock.tick(RENDER_FPS)


# ── Parallel worker (module-level so pickle can reach it on Windows) ────────────

def _eval_brain(args):
    """Evaluate one brain over EVAL_GAMES games; return (idx, averaged stats).
    agent.start() resets brain.score to 0 each game, so scores are accumulated manually.
    """
    idx, brain, level, max_turns, lowest_score_allowed, profile_name, scoring_mode = args
    apply_scoring_preset(scoring_mode)
    profile = get_profile(profile_name)

    levels_to_play = [1, 2, 3] if LEVEL_ROTATION else [level] * EVAL_GAMES

    totals = {k: 0.0 for k in ("score", "turns", "food_eaten", "score_survival",
                                "score_towards", "score_against", "score_ate", "score_bomb")}
    for lvl in levels_to_play:
        agent = Agent(lvl, max_turns, lowest_score_allowed, lambda: None, profile)
        agent.start(brain)          # resets brain.score = 0
        while not agent.done:
            agent.tick()
        totals["score"]          += brain.score
        totals["turns"]          += agent.turns
        totals["food_eaten"]     += agent.food_eaten
        totals["score_survival"] += agent.score_survival
        totals["score_towards"]  += agent.score_towards
        totals["score_against"]  += agent.score_against
        totals["score_ate"]      += agent.score_ate
        totals["score_bomb"]     += agent.score_bomb

    n = len(levels_to_play)
    brain.score = totals["score"] / n
    return idx, {k: v / n for k, v in totals.items()}


# ── Console helpers ────────────────────────────────────────────────────────────

def _banner(text: str):
    width = 72
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _sub(text: str):
    print(f"\n  ── {text}")


# ── Core training function (headless) ─────────────────────────────────────────

def run_training(seed_brains=None, run_tag="run", pool=None, visual_ctx=None) -> tuple:
    """
    Run GENERATIONS_PER_RUN generations without any display.
    pool: multiprocessing.Pool for parallel evaluation, or None for sequential.
    Returns (results: list[dict], sorted_brains: list, veteran_brain).
    """
    profile  = get_profile(PROFILE_NAME)
    n_inputs = profile_input_size(profile)

    seeds = [b for b in (seed_brains or []) if getattr(b, "input_size", None) == n_inputs]
    if seeds:
        # Derive rest of population from seeds via crossover+mutation — avoids the
        # "generation-1 crash" where 40 random brains drag the elite-median from 1600 to 700.
        n_extra   = max(0, GAMES_COUNT - len(seeds))
        raw_extra = [getOffspring(seeds, strategy=SELECTION_STRATEGY,
                                   power=SELECTION_POWER, k=TOURNAMENT_K)
                     for _ in range(n_extra)]
        fresh     = mutatePopulation(raw_extra, config.mutation_rate, config.mutation_amount)
    else:
        fresh = createPopulation(buildNetwork(n_inputs, 2), GAMES_COUNT)
    brains = seeds + fresh
    for b in brains:
        b.longevity = getattr(b, "longevity", 0)

    results       = []
    veteran_brain = None
    sorted_brains = brains[:]
    # temp_pool: list of {'brain': Network, 'count': int}
    # Each entry is a high-performing non-elite carried over unchanged for TEMP_PROMOTE_AFTER gens.
    temp_pool: list[dict] = []
    # Track the worst permanent-elite score from the PREVIOUS generation as the entry/survival
    # threshold. Non-elites always score ≤ the CURRENT worst elite by sorting definition, so we
    # compare against the PREVIOUS gen's bar — a brain that beats that bar shows real improvement.
    prev_worst_elite_score: float = float("-inf")
    # perm_elite_set: the STABLE permanent elite pool.
    # Set once in generation 0 by score ranking; from gen 1 onward it only changes
    # via explicit TempElite promotion (replace worst perm elite with promoted temp).
    perm_elite_set: list = []

    for gen in range(GENERATIONS_PER_RUN):

        # ── evaluate all brains (parallel or sequential) ───────────────────
        args = [
            (i, b, LEVEL, config.max_turns, config.lowest_score_allowed, PROFILE_NAME, SCORING_MODE)
            for i, b in enumerate(brains)
        ]
        raw = pool.map(_eval_brain, args) if pool is not None else [_eval_brain(a) for a in args]

        # write scores back to brain objects; build per-index stats lookup
        brain_stats: dict = {}
        for idx, stats in raw:
            brains[idx].score = stats["score"]
            brain_stats[idx]  = stats

        # ── collect stats ──────────────────────────────────────────────────
        sorted_brains = sortPopulation(brains)
        scores        = [b.score for b in brains]
        best_score    = sorted_brains[0].score
        worst_score   = sorted_brains[-1].score
        mean_score    = statistics.mean(scores)
        median_score  = statistics.median(scores)

        # Elite stats: use perm_elite_set scores (gen 0: not set yet, fall back to top-N)
        ref_elites       = sorted_brains[:ELITISM] if gen == 0 else perm_elite_set
        elite_scores     = [b.score for b in ref_elites]
        elite_median     = statistics.median(elite_scores)
        elite_best_score = max(elite_scores)

        turns_list   = [brain_stats[i]["turns"] for i in range(GAMES_COUNT)]
        best_turns   = round(max(turns_list))
        worst_turns  = round(min(turns_list))
        median_turns = statistics.median(turns_list)

        best_idx = next(i for i, b in enumerate(brains) if b is sorted_brains[0])
        ba       = brain_stats[best_idx]

        results.append({
            "generation"        : gen + 1,
            "best_score"        : round(best_score,           4),
            "mean_score"        : round(mean_score,           4),
            "median_score"      : round(median_score,         4),
            "worst_score"       : round(worst_score,          4),
            "best_turns"        : best_turns,
            "median_turns"      : round(median_turns,         1),
            "worst_turns"       : worst_turns,
            "best_food_eaten"   : ba["food_eaten"],
            "best_sc_survival"  : round(ba["score_survival"], 3),
            "best_sc_towards"   : round(ba["score_towards"],  3),
            "best_sc_against"   : round(ba["score_against"],  3),
            "best_sc_ate"       : round(ba["score_ate"],      3),
            "best_sc_bomb"      : round(ba["score_bomb"],     3),
            "elite_median_score": round(elite_median,         4),
        })

        print(
            f"  [{run_tag} {gen+1:3d}/{GENERATIONS_PER_RUN}]"
            f"  elite-best {elite_best_score:8.2f}  elite-median {elite_median:7.2f}"
            f"  turns {best_turns:5d}  food {ba['food_eaten']:4.1f}",
            flush=True,
        )

        if visual_ctx is not None:
            _run_visual_demo(
                sorted_brains[0],
                visual_ctx,
                f"[{run_tag}] Gen {gen+1}/{GENERATIONS_PER_RUN}  "
                f"best: {best_score:.1f}  elite-med: {elite_median:.1f}",
            )

        # ── GEN 0: establish permanent elite set by score ranking ──────────
        if gen == 0:
            perm_elite_set         = list(sorted_brains[:ELITISM])
            prev_worst_elite_score = perm_elite_set[-1].score
            for b in perm_elite_set:
                b.longevity = getattr(b, "longevity", 0) + 1
                if veteran_brain is None or b.longevity > veteran_brain.longevity:
                    veteran_brain = b

        # ── GEN 1+: perm_elite_set is fixed; only TempElite changes it ─────
        else:
            threshold = prev_worst_elite_score
            perm_ids  = set(id(b) for b in perm_elite_set)

            # evaluate existing temp elites: must beat prev-gen worst elite every round
            updated_temp: list[dict] = []
            for entry in temp_pool:
                b = entry["brain"]
                if id(b) in perm_ids:
                    continue  # runs as permanent elite already — drop from temp list
                if b.score > threshold:
                    entry["count"] += 1
                    if entry["count"] >= TEMP_PROMOTE_AFTER:
                        worst = min(perm_elite_set, key=lambda x: x.score)
                        perm_elite_set.remove(worst)
                        perm_elite_set.append(b)
                        perm_ids = set(id(x) for x in perm_elite_set)
                        if VERBOSE:
                            print(
                                f"  ↑  Temp-Elite promoted  (score {b.score:.2f}"
                                f" > threshold {threshold:.2f}  after {entry['count']} gens)",
                                flush=True,
                            )
                        # promoted — don't re-add to updated_temp
                    else:
                        updated_temp.append(entry)  # survived, try again
                # else: b.score ≤ threshold → eliminated

            # find new temp-elite candidates: any brain (not already perm or temp)
            # that beat the previous gen's worst permanent elite this round
            existing_ids = set(id(e["brain"]) for e in updated_temp)
            free_slots   = MAX_TEMP_ELITE - len(updated_temp)
            for b in sorted_brains:
                if free_slots <= 0:
                    break
                if id(b) not in perm_ids and id(b) not in existing_ids \
                        and b.score > threshold:
                    updated_temp.append({"brain": b, "count": 0})
                    existing_ids.add(id(b))
                    free_slots -= 1

            temp_pool = updated_temp

            # longevity: only permanent elites count
            for b in perm_elite_set:
                b.longevity = getattr(b, "longevity", 0) + 1
                if veteran_brain is None or b.longevity > veteran_brain.longevity:
                    veteran_brain = b

            # save threshold for the next generation
            prev_worst_elite_score = min(b.score for b in perm_elite_set)

        # ── evolve (skip on last generation) ──────────────────────────────
        if gen < GENERATIONS_PER_RUN - 1:
            temp_brains = [e["brain"] for e in temp_pool]
            n_offspring = GAMES_COUNT - len(perm_elite_set) - len(temp_brains)
            offspring   = [
                getOffspring(sorted_brains, strategy=SELECTION_STRATEGY,
                             power=SELECTION_POWER, k=TOURNAMENT_K)
                for _ in range(n_offspring)
            ]
            mutated = mutatePopulation(offspring, config.mutation_rate, config.mutation_amount)
            for b in mutated:
                b.longevity = 0
            brains = perm_elite_set + temp_brains + mutated

    return results, sorted_brains, veteran_brain


# ── Stats helpers ──────────────────────────────────────────────────────────────

def tail_stats(results: list, n: int = WINDOW) -> tuple:
    """Return (max_best_score, median_of_median_scores) over the last n generations."""
    tail = results[-n:]
    return (
        max(r["best_score"] for r in tail),
        statistics.median(
            [r["elite_median_score"] for r in tail]
        ),
    )


# ── Persistence ────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "generation",
    "best_score", "mean_score", "median_score",
    "elite_median_score", "worst_score",
    "best_turns", "median_turns", "worst_turns",
    "best_food_eaten",
    "best_sc_survival", "best_sc_towards", "best_sc_against",
    "best_sc_ate", "best_sc_bomb",
]


def write_csv(results: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def save_checkpoint(veteran, elite_brains: list, csv_results: list, label: str = ""):
    """Overwrite the rolling checkpoint so crashes never lose more than one run's progress."""
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(_CHECKPOINT_DIR, "veteran.pkl"), "wb") as f:
        pickle.dump({
            "network"      : veteran,
            "profile_name" : PROFILE_NAME,
            "longevity"    : getattr(veteran, "longevity", 0),
        }, f)
    with open(os.path.join(_CHECKPOINT_DIR, "elite.pkl"), "wb") as f:
        pickle.dump({"brains": elite_brains, "profile_name": PROFILE_NAME}, f)
    write_csv(csv_results, os.path.join(_CHECKPOINT_DIR, "training_report.csv"))
    tag = f"  [{label}]" if label else ""
    print(f"  💾 Checkpoint updated{tag} → {_CHECKPOINT_DIR}", flush=True)


def _append_progress(results: list, gen_offset: int) -> None:
    """Append one run's results to the cumulative progress CSV with shifted generation numbers.
    Bootstrap writes gens 1-300 (offset=0), first evo improvement writes 301-600 (offset=300), etc.
    """
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    write_header = not os.path.isfile(_PROGRESS_PATH)
    with open(_PROGRESS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in results:
            shifted = dict(row)
            shifted["generation"] = row["generation"] + gen_offset
            writer.writerow(shifted)


def show_eternity_graphs(progress_path: str = None):
    """Plot cumulative training progress from the progress CSV (same 4-panel layout as train.py).
    Dashed vertical lines mark each successful phase boundary (every GENERATIONS_PER_RUN gens).
    """
    import matplotlib.pyplot as plt

    path = progress_path or _PROGRESS_PATH
    if not os.path.isfile(path):
        print("  No progress data to plot yet.")
        return

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("  Progress CSV is empty.")
        return

    def _f(key):
        return [float(r.get(key) or 0) for r in rows]

    gens          = _f("generation")
    best_scores   = _f("best_score")
    mean_scores   = _f("mean_score")
    median_scores = _f("median_score")
    elite_medians = _f("elite_median_score")
    worst_scores  = _f("worst_score")
    best_turns    = _f("best_turns")
    median_turns  = _f("median_turns")
    worst_turns   = _f("worst_turns")
    food_eaten    = _f("best_food_eaten")
    sc_survival   = _f("best_sc_survival")
    sc_towards    = _f("best_sc_towards")
    sc_against    = _f("best_sc_against")
    sc_ate        = _f("best_sc_ate")
    sc_bomb       = _f("best_sc_bomb")

    max_gen     = max(gens) if gens else 0
    phase_lines = list(range(GENERATIONS_PER_RUN, int(max_gen) + 1, GENERATIONS_PER_RUN))

    def _phase_markers(ax):
        for i, x in enumerate(phase_lines):
            ax.axvline(x, color="white", linewidth=0.6, alpha=0.25, linestyle="--")
            ax.text(x + 1, ax.get_ylim()[1] * 0.97, f"P{i + 2}",
                    color="white", fontsize=6, alpha=0.4, va="top")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Bomberman Snake — Eternity Progress (cumulative)", fontsize=13, fontweight="bold")
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Score overview ────────────────────────────────────────────────────
    ax1.plot(gens, best_scores,   color="#00e676", linewidth=1.5, label="Best")
    ax1.plot(gens, median_scores, color="#ffd740", linewidth=1.5, label="Median")
    ax1.plot(gens, elite_medians, color="#b39ddb", linewidth=1.0, linestyle="--", label="Elite median", alpha=0.8)
    ax1.plot(gens, mean_scores,   color="#40c4ff", linewidth=1.0, linestyle="--", label="Mean", alpha=0.6)
    ax1.plot(gens, worst_scores,  color="#ff5252", linewidth=1.0, label="Worst", alpha=0.6)
    ax1.fill_between(gens, best_scores, worst_scores, alpha=0.07, color="white")
    ax1.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax1.set_title("Score (population)", fontsize=10)
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Turns + food ──────────────────────────────────────────────────────
    ax2.plot(gens, best_turns,   color="#00e676", linewidth=1.5, label="Best turns")
    ax2.plot(gens, median_turns, color="#ffd740", linewidth=1.5, label="Median turns")
    ax2.plot(gens, worst_turns,  color="#ff5252", linewidth=1.0, label="Worst turns", alpha=0.6)
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

    # ── Score gains ───────────────────────────────────────────────────────
    ax3.plot(gens, sc_survival, color="#40c4ff", linewidth=1.5, label="Survival")
    ax3.plot(gens, sc_towards,  color="#00e676", linewidth=1.5, label="→ Food (towards)")
    ax3.plot(gens, sc_ate,      color="#ffd740", linewidth=1.5, label="Ate food")
    ax3.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax3.set_title("Score breakdown — gains (best agent)", fontsize=10)
    ax3.set_ylabel("Score contribution")
    ax3.set_xlabel("Generation")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ── Score penalties ───────────────────────────────────────────────────
    ax4.plot(gens, sc_against, color="#ff5252", linewidth=1.5, label="← Food (away)")
    ax4.plot(gens, sc_bomb,    color="#ff9800", linewidth=1.5, label="Bomb penalty")
    ax4.axhline(0, color="white", linewidth=0.4, alpha=0.35)
    ax4.set_title("Score breakdown — penalties (best agent)", fontsize=10)
    ax4.set_ylabel("Score contribution")
    ax4.set_xlabel("Generation")
    ax4.legend(loc="lower left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    # Phase markers (after all axes are drawn so ylim is set)
    for ax in (ax1, ax2, ax3, ax4):
        _phase_markers(ax)

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")


_SUMMARY_FIELDS = [
    "rank", "run_index", "timestamp", "final_elite_median",
    "successful_improvements", "evo_rounds", "boot_attempts", "folder",
]


def update_run_summary(entry: dict):
    """Append entry to the global summary CSV, re-sort by final_elite_median desc, rewrite with updated ranks."""
    rows = []
    if os.path.isfile(_SUMMARY_PATH):
        with open(_SUMMARY_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "run_index"             : int(row["run_index"]),
                    "timestamp"             : row["timestamp"],
                    "final_elite_median": float(row["final_elite_median"]),
                    "successful_improvements": int(row["successful_improvements"]),
                    "evo_rounds"            : int(row["evo_rounds"]),
                    "boot_attempts"         : int(row["boot_attempts"]),
                    "folder"                : row["folder"],
                })

    rows.append({k: v for k, v in entry.items() if k != "rank"})
    rows.sort(key=lambda r: r["final_elite_median"], reverse=True)

    with open(_SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({**row, "rank": rank})

    print(f"       summary   : {_SUMMARY_PATH}")


def save_eternity_package(
    run_index: int,
    veteran,
    elite_brains: list,
    csv_results: list,
    final_elite_median: float,
    evo_rounds: int,
    successful_improvements: int,
    boot_attempts: int,
):
    """Write veteran + seeds + CSV into a timestamped runs/eternity sub-folder, then update summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder    = os.path.join(ETERNITY_DIR, f"{timestamp}_run{run_index:03d}")
    os.makedirs(folder, exist_ok=True)

    vet_path = os.path.join(folder, f"veteran_{timestamp}.pkl")
    with open(vet_path, "wb") as f:
        pickle.dump({
            "network"      : veteran,
            "profile_name" : PROFILE_NAME,
            "longevity"    : getattr(veteran, "longevity", 0),
        }, f)

    seed_path = os.path.join(folder, f"elite_{timestamp}.pkl")
    with open(seed_path, "wb") as f:
        pickle.dump({"brains": elite_brains, "profile_name": PROFILE_NAME}, f)

    csv_path = os.path.join(folder, f"training_report_{timestamp}.csv")
    write_csv(csv_results, csv_path)

    progress_dest = os.path.join(folder, "progress.csv")
    if os.path.isfile(_PROGRESS_PATH):
        shutil.copy2(_PROGRESS_PATH, progress_dest)

    print(f"\n  ★  Package saved → {folder}")
    print(f"       veteran   : {os.path.basename(vet_path)}")
    print(f"       elite     : {os.path.basename(seed_path)}")
    print(f"       report    : {os.path.basename(csv_path)}")
    if os.path.isfile(_PROGRESS_PATH):
        print("       progress  : progress.csv")

    update_run_summary({
        "run_index"              : run_index,
        "timestamp"              : timestamp,
        "final_elite_median": round(final_elite_median, 4),
        "successful_improvements": successful_improvements,
        "evo_rounds"             : evo_rounds,
        "boot_attempts"          : boot_attempts,
        "folder"                 : os.path.basename(folder),
    })

    return folder


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(ETERNITY_DIR, exist_ok=True)
    run_index  = 0
    n_workers  = N_WORKERS if N_WORKERS != 1 else None  # 1 → sequential, else pool
    pool_ctx   = Pool(n_workers) if n_workers != 1 else None
    visual_ctx = None

    if VISUAL:
        import pygame
        pygame.init()
        _win_w = FIELDSIZE * _V_CELL_PX
        _win_h = FIELDSIZE * _V_CELL_PX + _V_HEADER_H
        _surf  = pygame.display.set_mode((_win_w, _win_h))
        pygame.display.set_caption("Bomberman Snake — Eternity (Best Brain)")
        visual_ctx = {
            "surface": _surf,
            "font"   : pygame.font.SysFont(None, 18),
            "clock"  : pygame.time.Clock(),
        }

    print(f"Eternity runner started  —  output folder: {ETERNITY_DIR}")
    print(f"Config: {GAMES_COUNT} agents · {GENERATIONS_PER_RUN} gens/run · "
          f"profile={PROFILE_NAME} · level={LEVEL}  |  visual={VISUAL}")
    print(f"Workers: {n_workers or 'all cores'}  |  "
          f"Bootstrap threshold: best > {BOOTSTRAP_THRESHOLD}  |  "
          f"Improvement: +{(IMPROVEMENT_FACTOR-1)*100:.0f}% median  |  "
          f"Max failures: {MAX_FAILURES}")

    try:
      while True:
        run_index += 1
        gen_offset = 0
        # Fresh progress CSV for each eternity run
        if os.path.isfile(_PROGRESS_PATH):
            os.remove(_PROGRESS_PATH)

        _banner(f"ETERNITY RUN #{run_index}")

        # ── Phase 1: Bootstrap ─────────────────────────────────────────────
        boot_attempt = 0
        while True:
            boot_attempt += 1
            _sub(f"Bootstrap attempt #{boot_attempt}")
            results, sorted_brains, veteran = run_training(
                seed_brains=None,
                run_tag=f"Boot{boot_attempt}",
                pool=pool_ctx,
                visual_ctx=visual_ctx,
            )
            max_best, boot_median = tail_stats(results)
            print(
                f"\n  └ last-{WINDOW} best: {max_best:.2f}  "
                f"(threshold: {BOOTSTRAP_THRESHOLD})  "
                f"median: {boot_median:.2f}"
            )
            if max_best > BOOTSTRAP_THRESHOLD:
                print("  ✓  Bootstrap passed!")
                save_checkpoint(veteran, sorted_brains[:max(ELITISM, 1)], results,
                                label=f"run{run_index} bootstrap")
                _append_progress(results, gen_offset)
                gen_offset += GENERATIONS_PER_RUN
                break
            print("  ✗  Bootstrap failed — restarting from scratch")

        # store state after successful bootstrap
        current_seed          = sorted_brains[:max(ELITISM, 1)]
        prev_median           = boot_median
        best_veteran          = veteran
        last_good_results     = results
        last_good_elite       = list(current_seed)
        failure_count         = 0
        evo_round             = 0
        successful_improvements = 0

        print(f"\n  Bootstrap median (last {WINDOW} gens): {prev_median:.2f}")
        _sub("Entering evolution phase")

        # ── Phase 2: Evolution loop ────────────────────────────────────────
        while failure_count < MAX_FAILURES:
            evo_round += 1
            # For positive medians: need +5%. For negative: need /1.05 (less negative).
            target = prev_median * IMPROVEMENT_FACTOR if prev_median >= 0 else prev_median / IMPROVEMENT_FACTOR

            _sub(
                f"Evolution round #{evo_round}  |  "
                f"failures {failure_count}/{MAX_FAILURES}  |  "
                f"prev median {prev_median:.2f}  →  target ≥ {target:.2f}"
            )

            results, sorted_brains, veteran = run_training(
                seed_brains=current_seed,
                run_tag=f"Evo{evo_round:02d}",
                pool=pool_ctx,
                visual_ctx=visual_ctx,
            )
            _, new_median = tail_stats(results)

            change_pct = (new_median / prev_median - 1) * 100 if prev_median != 0 else 0
            print(
                f"\n  └ new median: {new_median:.2f}  "
                f"(prev: {prev_median:.2f}  target: {target:.2f}  "
                f"change: {change_pct:+.1f}%)"
            )

            if new_median >= target:
                print(
                    f"  ✓  Improved!  {prev_median:.2f} → {new_median:.2f}"
                    f"  (+{change_pct:.1f}%)  |  failures reset to 0"
                )
                failure_count           = 0
                prev_median             = new_median
                current_seed            = sorted_brains[:max(ELITISM, 1)]
                best_veteran            = veteran
                last_good_results       = results
                last_good_elite         = list(current_seed)
                successful_improvements += 1
                _append_progress(results, gen_offset)
                gen_offset += GENERATIONS_PER_RUN
                save_checkpoint(veteran, current_seed, results,
                                label=f"run{run_index} evo#{evo_round}")
            else:
                failure_count += 1
                print(
                    f"  ✗  No improvement  "
                    f"({new_median:.2f} < {target:.2f})  |  "
                    f"fail {failure_count}/{MAX_FAILURES}  "
                    f"— retrying with same seed"
                )
                # current_seed stays the same; retry same run

        # ── 10 failures → save package and restart ─────────────────────────
        _banner(f"RUN #{run_index} COMPLETE — {MAX_FAILURES} consecutive failures reached")
        print(f"  Best achieved median : {prev_median:.2f}")
        print(f"  Evolution rounds     : {evo_round}")
        print("  Saving Eternity package ...")
        save_eternity_package(
            run_index, best_veteran, last_good_elite, last_good_results,
            final_elite_median      = prev_median,
            evo_rounds              = evo_round,
            successful_improvements = successful_improvements,
            boot_attempts           = boot_attempt,
        )
        show_eternity_graphs()
        print("\n  Restarting from scratch (Phase 1)...")

    finally:
        if pool_ctx is not None:
            pool_ctx.terminate()
            pool_ctx.join()
        if visual_ctx is not None:
            import pygame
            pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — showing progress graphs...")
        show_eternity_graphs()

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
        Save Eternity package to  Eternity-Run/<timestamp>/
        Restart from Phase 1.

Outputs per saved package (inside Eternity-Run/<timestamp>_run<N>/):
    veteran_<timestamp>.pkl
    elite_<timestamp>.pkl
    training_report_<timestamp>.csv   (from the last improving run)
"""

import sys
import os
import csv
import pickle
import statistics
from datetime import datetime
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evolution"))

from network import buildNetwork
from evolution import sortPopulation, getOffspring, mutatePopulation, createPopulation
from constants import config
from agent import Agent
from profiles import get_profile, profile_input_size

# ── Config ─────────────────────────────────────────────────────────────────────

GAMES_COUNT          = 50
GENERATIONS_PER_RUN  = 300
LEVEL                = 1
PROFILE_NAME         = "full"
SELECTION_STRATEGY   = "power"   # "power" or "tournament"
SELECTION_POWER      = 4         # bias exponent for power strategy (higher = stronger top-bias)
TOURNAMENT_K         = 5         # candidates drawn per tournament (higher = more selective)
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
TEMP_PROMOTE_AFTER = 3     # how many consecutive good runs before promotion to permanent elite

# Set True to print Temp-Elite promotions and other internal events
VERBOSE = False

_HERE          = os.path.dirname(os.path.abspath(__file__))
ETERNITY_DIR   = os.path.join(_HERE, "Eternity-Run")
_SUMMARY_PATH  = os.path.join(ETERNITY_DIR, "eternity_summary.csv")
_CHECKPOINT_DIR = os.path.join(ETERNITY_DIR, "checkpoint")

ELITISM = round(ELITISM_RATE * GAMES_COUNT)


# ── Parallel worker (module-level so pickle can reach it on Windows) ────────────

def _eval_brain(args):
    """Evaluate one brain over EVAL_GAMES games; return (idx, averaged stats).
    agent.start() resets brain.score to 0 each game, so scores are accumulated manually.
    """
    idx, brain, level, max_turns, lowest_score_allowed, profile_name = args
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

def run_training(seed_brains=None, run_tag="run", pool=None) -> tuple:
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
            (i, b, LEVEL, config.max_turns, config.lowest_score_allowed, PROFILE_NAME)
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
    """Write veteran + seeds + CSV into a timestamped Eternity-Run sub-folder, then update summary."""
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

    print(f"\n  ★  Package saved → {folder}")
    print(f"       veteran   : {os.path.basename(vet_path)}")
    print(f"       elite     : {os.path.basename(seed_path)}")
    print(f"       report    : {os.path.basename(csv_path)}")

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

    print(f"Eternity runner started  —  output folder: {ETERNITY_DIR}")
    print(f"Config: {GAMES_COUNT} agents · {GENERATIONS_PER_RUN} gens/run · "
          f"profile={PROFILE_NAME} · level={LEVEL}")
    print(f"Workers: {n_workers or 'all cores'}  |  "
          f"Bootstrap threshold: best > {BOOTSTRAP_THRESHOLD}  |  "
          f"Improvement: +{(IMPROVEMENT_FACTOR-1)*100:.0f}% median  |  "
          f"Max failures: {MAX_FAILURES}")

    try:
      while True:
        run_index += 1
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
            )
            max_best, boot_median = tail_stats(results)
            print(
                f"\n  └ last-{WINDOW} best: {max_best:.2f}  "
                f"(threshold: {BOOTSTRAP_THRESHOLD})  "
                f"median: {boot_median:.2f}"
            )
            if max_best > BOOTSTRAP_THRESHOLD:
                print(f"  ✓  Bootstrap passed!")
                save_checkpoint(veteran, sorted_brains[:max(ELITISM, 1)], results,
                                label=f"run{run_index} bootstrap")
                break
            print(f"  ✗  Bootstrap failed — restarting from scratch")

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
        print(f"  Saving Eternity package ...")
        save_eternity_package(
            run_index, best_veteran, last_good_elite, last_good_results,
            final_elite_median      = prev_median,
            evo_rounds              = evo_round,
            successful_improvements = successful_improvements,
            boot_attempts           = boot_attempt,
        )
        print(f"\n  Restarting from scratch (Phase 1)...")

    finally:
        if pool_ctx is not None:
            pool_ctx.terminate()
            pool_ctx.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — exiting.")

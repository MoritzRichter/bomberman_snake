"""
tuner.py — Coarse-to-fine hyperparameter search for the genetic algorithm.

Two independent searches:

  search()            — profile, score params, urgency weights (numeric, refinable)
  search_selection()  — selection strategy + elitism rate (categorical)

Run both with:
    python tuner.py

Or import and call individually:
    from tuner import Tuner
    Tuner().search()
    Tuner().search_selection()

Results are saved to runs/tuner/tuner_<search>_<timestamp>.csv.
"""
import sys
import os
import csv
import copy
import time
import itertools
import statistics
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evolution'))

from network import buildNetwork
from evolution import sortPopulation, getOffspring, mutatePopulation, createPopulation
from selection import STRATEGIES
from agent import Agent
from profiles import PROFILES, profile_input_size
from constants import config as G

TUNER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "tuner")


# ─────────────────────────────────────────────────────────────────────────────
# Trial configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialConfig:
    """One point in the hyperparameter space."""

    # Sensor profile
    profile_name:        str   = "bomb_aware"
    urgency_weight:      float = 2.0    # weight for food_timer / bomb_timer sensors

    # Score parameters
    score_towards_food:  float = 1.0
    score_against_food:  float = -0.5
    score_ate_food:      float = 2.0
    score_survived_tick: float = 0.01   # reward per tick alive
    score_bomb_exploded: float = 0   # penalty when bomb explodes

    # Evolution parameters
    mutation_rate:       float = 0.5
    selection_power:     int   = 4      # only used by "power" strategy

    # Selection
    selection_strategy:  str   = "power"
    elitism_rate:        float = 0.2    # fraction of population kept unchanged
    tournament_k:        int   = 3      # only used by "tournament" strategy
    top_n_ratio:         float = 0.2    # only used by "top_n" strategy

    def build_profile(self) -> dict:
        base = copy.deepcopy(PROFILES[self.profile_name])
        for key in ("food_timer", "bomb_timer"):
            if base.get(key, 0) > 0:
                base[key] = self.urgency_weight
        return base

    def selection_kwargs(self) -> dict:
        """Strategy-specific kwargs for getOffspring."""
        if self.selection_strategy == "power":
            return {"power": self.selection_power}
        if self.selection_strategy == "tournament":
            return {"k": self.tournament_k}
        if self.selection_strategy == "top_n":
            return {"top_ratio": self.top_n_ratio}
        return {}

    def as_dict(self) -> dict:
        return {
            "profile":            self.profile_name,
            "urgency_weight":     round(self.urgency_weight, 4),
            "score_towards":      self.score_towards_food,
            "score_against":      round(self.score_against_food, 4),
            "score_ate":          round(self.score_ate_food, 4),
            "score_survived_tick": round(self.score_survived_tick, 5),
            "score_bomb_exploded": round(self.score_bomb_exploded, 4),
            "mutation_rate":      round(self.mutation_rate, 4),
            "selection_power":    self.selection_power,
            "selection_strategy": self.selection_strategy,
            "elitism_rate":       round(self.elitism_rate, 3),
            "tournament_k":       self.tournament_k,
            "top_n_ratio":        self.top_n_ratio,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tuner
# ─────────────────────────────────────────────────────────────────────────────

class Tuner:
    """
    search()           → coarse-to-fine over profile + score params
    search_selection() → flat grid over all selection strategies + elitism rates
    """

    BOUNDS = {
        "urgency_weight":     (0.1, 6.0),
        "score_against_food": (-5.0, -0.1),
        "score_ate_food":     (0.5, 10.0),
        "mutation_rate":      (0.1, 0.95),
    }
    INT_BOUNDS = {
        "selection_power": (2, 10),
    }

    def __init__(
        self,
        rounds:            int   = 3,
        top_k:             int   = 5,
        trial_generations: int   = 15,
        trial_agents:      int   = 30,
        trial_max_turns:   int   = 500,
        level:             int   = 1,
    ):
        self.rounds            = rounds
        self.top_k             = top_k
        self.trial_generations = trial_generations
        self.trial_agents      = trial_agents
        self.trial_max_turns   = trial_max_turns
        self.level             = level
        self._all_results: list[dict] = []

    # ── Search 1: profile + score params ─────────────────────────────────────

    def search(self):
        """Coarse-to-fine search over sensor profiles and score/mutation parameters."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(TUNER_DIR, f"tuner_params_{timestamp}.csv")

        self._header("Profile & Score Parameter Search", csv_path)

        step    = 1.0
        configs = self._coarse_configs()
        top     = self._run_round(configs, label="Coarse", step=step)

        for r in range(2, self.rounds + 1):
            step   /= 2
            configs = self._refine_configs([tc for _, tc in top], step)
            top     = self._run_round(configs, label=f"Refine {r-1}", step=step)

        self._save_csv(csv_path)
        self._print_summary(top, csv_path)

    # ── Search 2: selection strategy + elitism ────────────────────────────────

    def search_selection(self):
        """
        Flat grid search over all selection strategies and elitism rates.
        Also varies strategy-specific parameters (tournament k, top_n ratio, power).

        Strategies tested:
          power      — x^k distribution; tests k in [2, 4, 8]
          tournament — random k-best; tests k in [2, 3, 5, 10]
          roulette   — fitness-proportional probability
          top_n      — uniform from top n%; tests n in [10%, 20%, 30%]
          random     — uniform random (control baseline)

        Elitism rates: 0% / 10% / 20% / 30%
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(TUNER_DIR, f"tuner_selection_{timestamp}.csv")

        self._header("Selection Strategy Search", csv_path)
        configs = self._selection_configs()
        top     = self._run_round(configs, label="Selection grid", step=0)

        self._save_csv(csv_path)
        self._print_selection_summary(top, csv_path)

    # ── Config generation: profile + score ───────────────────────────────────

    def _coarse_configs(self) -> list[TrialConfig]:
        raw = itertools.product(
            ["basic", "bomb_aware", "timer", "full"],
            [0.5, 1.0, 2.0, 3.0],
            [-0.5, -1.0, -1.5, -2.0],
            [1.0, 2.0, 4.0],
        )
        seen, configs = set(), []
        for profile_name, uw, af, ate in raw:
            has_timers = any(PROFILES[profile_name].get(k, 0) > 0 for k in ("food_timer", "bomb_timer"))
            key = (profile_name, uw if has_timers else None, af, ate)
            if key in seen:
                continue
            seen.add(key)
            configs.append(TrialConfig(profile_name=profile_name, urgency_weight=uw,
                                       score_against_food=af, score_ate_food=ate))
        return configs

    def _refine_configs(self, best: list[TrialConfig], step: float) -> list[TrialConfig]:
        seen, configs = set(), []

        def _key(tc):
            return (tc.profile_name, tc.selection_strategy,
                    round(tc.urgency_weight, 5), round(tc.score_against_food, 5),
                    round(tc.score_ate_food, 5), round(tc.mutation_rate, 5),
                    tc.selection_power)

        def _add(tc):
            k = _key(tc)
            if k not in seen:
                seen.add(k)
                configs.append(tc)

        for base in best:
            _add(base)
            for param, (lo, hi) in self.BOUNDS.items():
                for delta in (-step, +step):
                    tc = copy.copy(base)
                    setattr(tc, param, max(lo, min(hi, round(getattr(base, param) + delta, 6))))
                    _add(tc)
            int_step = max(1, round(step * 2))
            for param, (lo, hi) in self.INT_BOUNDS.items():
                for delta in (-int_step, +int_step):
                    tc = copy.copy(base)
                    setattr(tc, param, max(lo, min(hi, int(getattr(base, param)) + delta)))
                    _add(tc)

        return configs

    # ── Config generation: selection ──────────────────────────────────────────

    def _selection_configs(self) -> list[TrialConfig]:
        """All strategy × elitism × strategy-specific parameter combinations."""
        configs = []
        elitism_rates = [0.0, 0.1, 0.2, 0.3]

        for elitism in elitism_rates:
            # power: vary selection_power
            for power in [2, 4, 8]:
                configs.append(TrialConfig(selection_strategy="power",
                                           selection_power=power, elitism_rate=elitism))

            # tournament: vary k
            for k in [2, 3, 5, 10]:
                configs.append(TrialConfig(selection_strategy="tournament",
                                           tournament_k=k, elitism_rate=elitism))

            # roulette: no extra param
            configs.append(TrialConfig(selection_strategy="roulette", elitism_rate=elitism))

            # top_n: vary ratio
            for ratio in [0.1, 0.2, 0.3]:
                configs.append(TrialConfig(selection_strategy="top_n",
                                           top_n_ratio=ratio, elitism_rate=elitism))

            # random: control baseline
            configs.append(TrialConfig(selection_strategy="random", elitism_rate=elitism))

        return configs

    # ── Single trial ──────────────────────────────────────────────────────────

    def _run_trial(self, tc: TrialConfig) -> float:
        profile    = tc.build_profile()
        n_inputs   = profile_input_size(profile)
        n_elitists = max(0, round(tc.elitism_rate * self.trial_agents))
        n_offspring = self.trial_agents - n_elitists

        saved = (G.score.points_towards_food, G.score.points_against_food,
                 G.score.points_ate_food, G.score.points_survived_tick,
                 G.score.points_bomb_exploded, G.mutation_rate, G.selection_power, G.max_turns)
        G.score.points_towards_food   = tc.score_towards_food
        G.score.points_against_food   = tc.score_against_food
        G.score.points_ate_food       = tc.score_ate_food
        G.score.points_survived_tick  = tc.score_survived_tick
        G.score.points_bomb_exploded  = tc.score_bomb_exploded
        G.mutation_rate               = tc.mutation_rate
        G.selection_power             = tc.selection_power
        G.max_turns                   = self.trial_max_turns

        try:
            brains = createPopulation(buildNetwork(n_inputs, 2), self.trial_agents)
            agents = [
                Agent(level=self.level, max_turns=self.trial_max_turns,
                      lowest_score_allowed=G.lowest_score_allowed,
                      on_game_over=lambda: None, profile=profile)
                for _ in range(self.trial_agents)
            ]

            best_per_gen: list[float] = []

            for gen in range(self.trial_generations):
                for i, agent in enumerate(agents):
                    agent.start(brains[i])
                while not all(a.done for a in agents):
                    for a in agents:
                        a.tick()

                sorted_brains = sortPopulation(brains)
                best_per_gen.append(sorted_brains[0].score)

                if gen < self.trial_generations - 1:
                    elitists  = sorted_brains[:n_elitists]
                    offspring = [
                        getOffspring(sorted_brains, strategy=tc.selection_strategy,
                                     **tc.selection_kwargs())
                        for _ in range(n_offspring)
                    ]
                    brains = elitists + mutatePopulation(offspring, tc.mutation_rate, G.mutation_amount)

            return statistics.mean(best_per_gen[-5:])

        finally:
            (G.score.points_towards_food, G.score.points_against_food,
             G.score.points_ate_food, G.score.points_survived_tick,
             G.score.points_bomb_exploded, G.mutation_rate, G.selection_power, G.max_turns) = saved

    # ── Round runner ──────────────────────────────────────────────────────────

    def _run_round(self, configs: list[TrialConfig], label: str, step: float):
        print(f"{'─'*72}")
        print(f"  [{label}]  {len(configs)} trials" + (f"  step={step:.4f}" if step else ""))
        print(f"{'─'*72}")

        results: list[tuple[float, TrialConfig]] = []
        t_round = time.time()

        for i, tc in enumerate(configs):
            t0      = time.time()
            fitness = self._run_trial(tc)
            elapsed = time.time() - t0

            self._all_results.append({"phase": label, "fitness": round(fitness, 4), **tc.as_dict()})
            results.append((fitness, tc))

            print(
                f"  [{i+1:3d}/{len(configs)}]"
                f"  {tc.profile_name:10s}"
                f"  {tc.selection_strategy:10s}"
                f"  elite={tc.elitism_rate:.0%}"
                f"  uw={tc.urgency_weight:.1f}"
                f"  af={tc.score_against_food:+.2f}"
                f"  ate={tc.score_ate_food:.1f}"
                f"  →  {fitness:8.2f}"
                f"  ({elapsed:.1f}s)"
            )

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:self.top_k]
        total = time.time() - t_round

        print(f"\n  Top {self.top_k}  (took {total:.0f}s):")
        for rank, (fitness, tc) in enumerate(top, 1):
            print(f"    #{rank}  {fitness:8.2f}  "
                  f"profile={tc.profile_name}  strategy={tc.selection_strategy}  "
                  f"elite={tc.elitism_rate:.0%}  "
                  f"af={tc.score_against_food:+.2f}  ate={tc.score_ate_food:.1f}")
        print()
        return top

    # ── Persistence & summaries ───────────────────────────────────────────────

    def _save_csv(self, path: str):
        if not self._all_results:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fields = list(self._all_results[0].keys())
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
            csv.DictWriter(f, fieldnames=fields).writerows(self._all_results)
        # reset for next search
        self._all_results = []
        print(f"Results saved → {path}\n")

    def _header(self, title: str, csv_path: str):
        print(f"\n{'═'*72}")
        print(f"  {title}")
        print(f"  {self.trial_generations} gens  |  {self.trial_agents} agents  "
              f"|  max {self.trial_max_turns} turns  |  output → {csv_path}")
        print(f"{'═'*72}\n")

    def _print_summary(self, top, csv_path: str):
        print(f"{'═'*72}")
        print(f"  Profile search complete  →  {csv_path}")
        print(f"{'═'*72}")
        fitness, tc = top[0]
        print(f"\n  Best  (fitness={fitness:.2f}):")
        for k, v in tc.as_dict().items():
            print(f"    {k:22s}: {v}")
        print("\n  → Copy to train.py:")
        print(f"    PROFILE_NAME       = \"{tc.profile_name}\"")
        print(f"    SELECTION_STRATEGY = \"{tc.selection_strategy}\"")
        print(f"    ELITISM_RATE       = {tc.elitism_rate}")
        print("    # constants.py ScoreConfig:")
        print(f"    points_against_food = {tc.score_against_food}")
        print(f"    points_ate_food     = {tc.score_ate_food}")
        print("    # profiles.py (timer/full):")
        print(f"    food_timer = bomb_timer = {tc.urgency_weight}")
        print()

    def _print_selection_summary(self, top, csv_path: str):
        print(f"{'═'*72}")
        print(f"  Selection search complete  →  {csv_path}")
        print(f"{'═'*72}")
        print(f"\n  Strategy comparison (top {self.top_k}):")

        by_strategy: dict[str, list] = {s: [] for s in STRATEGIES}
        for fitness, tc in top:
            by_strategy[tc.selection_strategy].append(fitness)

        print(f"\n  {'Strategy':12s}  {'Best fitness':>14s}  {'Detail'}")
        print(f"  {'─'*60}")
        fitness, tc = top[0]
        for rank, (fitness, tc) in enumerate(top, 1):
            detail = ""
            if tc.selection_strategy == "power":
                detail = f"power={tc.selection_power}"
            elif tc.selection_strategy == "tournament":
                detail = f"k={tc.tournament_k}"
            elif tc.selection_strategy == "top_n":
                detail = f"ratio={tc.top_n_ratio:.0%}"
            print(f"  #{rank}  {tc.selection_strategy:10s}  "
                  f"fitness={fitness:8.2f}  elite={tc.elitism_rate:.0%}  {detail}")

        best_fitness, best_tc = top[0]
        print(f"\n  → Best: {best_tc.selection_strategy}"
              f"  (elite={best_tc.elitism_rate:.0%}"
              f"  fitness={best_fitness:.2f})")
        print("\n  → Copy to train.py:")
        print(f"    SELECTION_STRATEGY = \"{best_tc.selection_strategy}\"")
        print(f"    ELITISM_RATE       = {best_tc.elitism_rate}")
        if best_tc.selection_strategy == "power":
            print(f"    # constants.py: selection_power = {best_tc.selection_power}")
        elif best_tc.selection_strategy == "tournament":
            print(f"    # selection.py call: k={best_tc.tournament_k}")
        elif best_tc.selection_strategy == "top_n":
            print(f"    # selection.py call: top_ratio={best_tc.top_n_ratio}")
        print()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = Tuner(
        rounds            = 3,
        top_k             = 5,
        trial_generations = 15,
        trial_agents      = 30,
        trial_max_turns   = 500,
    )

    print("Running profile & score search...")
    t.search()

    print("Running selection strategy search...")
    t.search_selection()

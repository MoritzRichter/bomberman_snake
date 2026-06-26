import random
import math


# ─────────────────────────────────────────────────────────────────────────────
# Selection strategies
# ─────────────────────────────────────────────────────────────────────────────
#
# All strategies receive a population sorted best-first (index 0 = highest score).
#
# POWER      — x^power skew: high power → almost always top, power=1 → uniform random
# TOURNAMENT — pick k candidates at random, return the best
# ROULETTE   — fitness-proportional probability (scores shifted to be non-negative)
# TOP_N      — uniform random from the top n% of the population
# RANDOM     — uniform random (baseline / control)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = ("power", "tournament", "roulette", "top_n", "random")


def power_selection(population, power: int = 4):
    """Bias toward top: index = floor(U[0,1]^power * N). Higher power = stronger bias."""
    index = math.floor((random.random() ** power) * len(population))
    return population[index]


def tournament_selection(population, k: int = 3):
    """Pick k candidates uniformly at random, return the one with the highest score."""
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda n: n.score if n.score is not None else float("-inf"))


def roulette_selection(population):
    """
    Fitness-proportional selection.
    Scores are shifted by the minimum so they are all non-negative before sampling.
    If all scores are equal, falls back to uniform random.
    """
    scores = [n.score if n.score is not None else 0.0 for n in population]
    min_s  = min(scores)
    shifted = [s - min_s for s in scores]
    total   = sum(shifted)

    if total == 0:
        return random.choice(population)

    r, cumulative = random.uniform(0, total), 0.0
    for brain, s in zip(population, shifted):
        cumulative += s
        if cumulative >= r:
            return brain
    return population[-1]


def top_n_selection(population, top_ratio: float = 0.2):
    """Uniform random from the top top_ratio fraction of the population."""
    n = max(1, math.ceil(top_ratio * len(population)))
    return random.choice(population[:n])


def random_selection(population):
    """Uniform random across the entire population (control baseline)."""
    return random.choice(population)


# ─────────────────────────────────────────────────────────────────────────────
# Unified interface
# ─────────────────────────────────────────────────────────────────────────────

def select_parent(population, strategy: str = "power", **kwargs):
    """
    Select one parent from the sorted population using the given strategy.

    Args:
        population:  list of Networks sorted best-first
        strategy:    one of STRATEGIES
        **kwargs:    strategy-specific parameters:
                       power      → power (int, default 4)
                       tournament → k (int, default 3)
                       top_n      → top_ratio (float, default 0.2)
    """
    if strategy == "power":
        return power_selection(population, kwargs.get("power", 4))
    if strategy == "tournament":
        return tournament_selection(population, kwargs.get("k", 3))
    if strategy == "roulette":
        return roulette_selection(population)
    if strategy == "top_n":
        return top_n_selection(population, kwargs.get("top_ratio", 0.2))
    if strategy == "random":
        return random_selection(population)
    raise ValueError(f"Unknown selection strategy '{strategy}'. Choose from: {STRATEGIES}")


# Keep old name for backward compatibility
def powerSelection(population, selectionPower):
    return power_selection(population, selectionPower)

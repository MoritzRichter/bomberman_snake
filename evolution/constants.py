import math


def sigmoid(x) :
    return 1 / (1 + math.exp(-x))


def tanh(x) :
    return math.tanh(x)


def relu(x) :
    return max(0, x)


def identity(x) :
    return x


ALL_ACTIVATIONS = [sigmoid, tanh, relu, identity]






class WeightConfig :
    def __init__(self, min, max) :
        self.min = min
        self.max = max


class BiasConfig :
    def __init__(self, min, max) :
        self.min = min
        self.max = max


class ActivationFunctionConfig :
    def __init__(self, mutateOutput) :
        self.mutateOutput = mutateOutput


class SwapNodesConfig :
    def __init__(self, mutateOutput) :
        self.mutateOutput = mutateOutput


class MutationsConfig :
    def __init__(self) :
        self.connectionWeight      = WeightConfig(min=-0.1, max=0.1)   # small perturbation
        self.connectionWeightLarge = WeightConfig(min=-2.0, max=2.0)   # occasional full reset
        self.bias = BiasConfig(min=-0.5, max=0.5)
        self.activationFunction = ActivationFunctionConfig(mutateOutput=False)
        self.swapNodes = SwapNodesConfig(mutateOutput=False)


class ScoreConfig :
    def __init__(self) :
        self.points_towards_food    = 1.0
        self.points_against_food    = -0.5
        self.points_ate_food        = 5.0
        self.points_survived_tick   = 0.03
        self.points_bomb_exploded   = -1.0
        self.points_per_length_tick = 0.0   # bonus per body segment per tick (length mode)


class Config :
    def __init__(self) :
        self.warnings  = False
        self.mutations = MutationsConfig()
        self.score     = ScoreConfig()

        # Population settings
        self.games_count          = 100
        self.max_turns            = 5000000
        self.lowest_score_allowed = -50

        # Evolution settings
        self.mutation_rate   = 0.5
        self.mutation_amount = 3
        self.elitism         = round(0.2 * self.games_count)  # top 20% carried over unchanged
        self.selection_power = 4  # higher = stronger bias toward top-ranked parents


config = Config()


# ── Scoring presets ────────────────────────────────────────────────────────────
# Each preset overrides all ScoreConfig fields.  Switch via apply_scoring_preset().

SCORING_PRESETS: dict[str, dict] = {

    # Standard-Modus: ausgewogenes Verhältnis aller Score-Komponenten
    "balanced": {
        "points_towards_food"    :  1.0,
        "points_against_food"    : -0.5,
        "points_ate_food"        :  5.0,
        "points_survived_tick"   :  0.03,
        "points_bomb_exploded"   : -1.0,
        "points_per_length_tick" :  0.0,
    },

    # Überleben: möglichst lange am Leben bleiben
    "survival": {
        "points_towards_food"    :  0.3,
        "points_against_food"    : -0.1,
        "points_ate_food"        :  1.0,
        "points_survived_tick"   :  1.15,
        "points_bomb_exploded"   : -1.0,
        "points_per_length_tick" :  0.0,
    },

    # Fresser: möglichst viel Essen in kürzester Zeit
    "food": {
        "points_towards_food"    :  3.0,
        "points_against_food"    : -2.0,
        "points_ate_food"        : 25.0,
        "points_survived_tick"   :  0.0,
        "points_bomb_exploded"   : -15.0,
        "points_per_length_tick" :  0.0,
    },

    # Länge: möglichst lange Schlange; jedes Segment gibt Bonus pro Tick
    "length": {
        "points_towards_food"    :  1.0,
        "points_against_food"    : -0.3,
        "points_ate_food"        : 10.0,
        "points_survived_tick"   :  0.0,
        "points_bomb_exploded"   : -5.0,
        "points_per_length_tick" :  0.02,  # × len(snake) pro Tick
    },

    # In Game Score: Einzige Fittingfunktion ist der tatsächliche in game score
    "in_game_score": {
        "points_towards_food"    : 0,
        "points_against_food"    : 0,
        "points_ate_food"        : 1,
        "points_survived_tick"   : 0,
        "points_bomb_exploded"   : 0,
        "points_per_length_tick" : 0,  # × len(snake) pro Tick
    },
}


def apply_scoring_preset(name: str) -> None:
    """Apply a named scoring preset to the global config object."""
    if name not in SCORING_PRESETS:
        raise ValueError(f"Unknown scoring preset '{name}'. Choose from: {list(SCORING_PRESETS)}")
    for attr, value in SCORING_PRESETS[name].items():
        setattr(config.score, attr, value)
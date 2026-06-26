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
        self.connectionWeight = WeightConfig(min=-0.5, max=0.5)
        self.bias = BiasConfig(min=-0.5, max=0.5)
        self.activationFunction = ActivationFunctionConfig(mutateOutput=False)
        self.swapNodes = SwapNodesConfig(mutateOutput=False)


class ScoreConfig :
    def __init__(self) :
        self.points_towards_food = 1
        self.points_against_food = -1.5
        self.points_ate_food     = 2


class Config :
    def __init__(self) :
        self.warnings  = True
        self.mutations = MutationsConfig()
        self.score     = ScoreConfig()

        # Population settings
        self.games_count          = 100
        self.max_turns            = 5000
        self.lowest_score_allowed = -50

        # Evolution settings
        self.mutation_rate   = 0.5
        self.mutation_amount = 3
        self.elitism         = round(0.2 * self.games_count)  # top 20% carried over unchanged
        self.selection_power = 4  # higher = stronger bias toward top-ranked parents


config = Config()
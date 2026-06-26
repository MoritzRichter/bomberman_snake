import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from network import buildNetwork
from agent import Agent
# from evolution import sortPopulation, getOffspring, mutatePopulation  # to be implemented
# from evolution import createPopulation                                 # to be implemented
from constants import config

# -------------------------------------------------------
# Network shape
# -------------------------------------------------------

INPUT_SIZE  = 9  # can_move x3 + is_food x3 + is_bomb x3
OUTPUT_SIZE = 2  # turn left, turn right


# -------------------------------------------------------
# Ecosystem
# -------------------------------------------------------

class Ecosystem :
    """Manages the full population of agents across generations."""

    def __init__(self, games_count, level, max_turns, lowest_score_allowed) :
        self.games_count          = games_count
        self.level                = level
        self.max_turns            = max_turns
        self.lowest_score_allowed = lowest_score_allowed

        self.games          = []
        self.brains         = []
        self.generation     = 0
        self.games_finished = 0


    # --- Setup -------------------------------------------------------------

    def initBrains(self) :
        self.brains = createPopulation(buildNetwork(INPUT_SIZE, OUTPUT_SIZE), self.games_count)

    def initGames(self) :
        self.games = []

        for i in range(self.games_count) :
            self.games.append(
                Agent(
                    level                = self.level,
                    max_turns            = self.max_turns,
                    lowest_score_allowed = self.lowest_score_allowed,
                    on_game_over         = lambda: self.endGeneration(),
                )
            )


    # --- Generation loop ---------------------------------------------------

    def startGeneration(self) :
        self.games_finished = 0

        for i, agent in enumerate(self.games) :
            agent.start(self.brains[i])

    def endGeneration(self) :
        # Wait until every agent has finished before evolving
        if self.games_finished + 1 < len(self.games) :
            self.games_finished += 1
            return

        brains_sorted  = sortPopulation(self.brains)
        elitists       = []
        new_generation = []

        for i in range(config.elitism) :
            elitists.append(brains_sorted[i])

        for i in range(self.games_count - config.elitism) :
            new_generation.append(getOffspring(brains_sorted))

        mutated_new_generation = mutatePopulation(
            new_generation,
            config.mutation_rate,
            config.mutation_amount,
        )

        self.brains     = elitists + mutated_new_generation
        self.generation += 1
        self.startGeneration()


    # --- Entry points ------------------------------------------------------

    def start(self) :
        self.initBrains()
        self.initGames()
        self.startGeneration()

    def stop(self) :
        self.games          = []
        self.brains         = []
        self.generation     = 0
        self.games_finished = 0

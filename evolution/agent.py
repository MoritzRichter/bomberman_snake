import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.logic import GameLogic
from network import Network, activate_network
from sensors import MoveHelper, TURN_LEFT, TURN_RIGHT
from constants import config


class Agent :
    """Owns one GameLogic instance and one brain. Runs one full game per start() call."""

    def __init__(self, level, max_turns, lowest_score_allowed, on_game_over, profile=None) :
        self.level                = level
        self.max_turns            = max_turns
        self.lowest_score_allowed = lowest_score_allowed
        self.on_game_over         = on_game_over
        self.profile              = profile

        self.game  = GameLogic(level)
        self.brain = None
        self.turns = 0
        self._done = False
        # Score component accumulators (reset each start())
        self.score_survival = 0.0
        self.score_towards  = 0.0
        self.score_against  = 0.0
        self.score_ate      = 0.0
        self.score_bomb     = 0.0
        self.food_eaten     = 0


    # --- Entry point -------------------------------------------------------

    def start(self, brain: Network) :
        """Initialise a new game. Call tick() externally each step."""
        self.brain          = brain
        self.brain.score    = 0
        self.turns          = 0
        self._done          = False
        self.score_survival = 0.0
        self.score_towards  = 0.0
        self.score_against  = 0.0
        self.score_ate      = 0.0
        self.score_bomb     = 0.0
        self.food_eaten     = 0
        self.game.reset()

    @property
    def done(self) -> bool :
        return self._done


    # --- Game loop ---------------------------------------------------------

    def tick(self) :
        if self._done :
            return

        helper  = MoveHelper(self.game, self.profile)
        inputs  = helper.get_inputs()
        outputs = activate_network(self.brain, inputs)

        turn_left  = round(outputs[0])
        turn_right = round(outputs[1])

        # Resolve intended absolute direction from network output
        if turn_left :
            direction     = TURN_LEFT[self.game.direction]
            intended_side = MoveHelper.LEFT
        elif turn_right :
            direction     = TURN_RIGHT[self.game.direction]
            intended_side = MoveHelper.RIGHT
        else :
            direction     = self.game.direction
            intended_side = MoveHelper.FORWARD

        # Score the chosen direction before the move happens
        self._update_score(helper.is_food(intended_side))

        had_explosion = self.game.explosion
        ate_food, died = self.game.step(int(direction))
        self.turns += 1

        if ate_food :
            pts = config.score.points_ate_food
            self._add_score(pts)
            self.score_ate  += pts
            self.food_eaten += 1

        # Penalise bomb explosions (snake could have eaten the food in time)
        if not had_explosion and self.game.explosion :
            pts = config.score.points_bomb_exploded
            self._add_score(pts)
            self.score_bomb += pts

        # Small reward for surviving each step
        pts = config.score.points_survived_tick
        self._add_score(pts)
        self.score_survival += pts

        # Length bonus: reward proportional to current snake size (length mode)
        if config.score.points_per_length_tick != 0.0 :
            pts = config.score.points_per_length_tick * len(self.game.snake)
            self._add_score(pts)
            self.score_survival += pts

        if self._is_over() :
            self._done = True


    # --- Scoring -----------------------------------------------------------

    def _update_score(self, is_food: bool) :
        """Reward moving toward food, penalise moving away."""
        if is_food :
            pts = config.score.points_towards_food
            self._add_score(pts)
            self.score_towards += pts
        else :
            pts = config.score.points_against_food
            self._add_score(pts)
            self.score_against += pts

    def _add_score(self, score: float) :
        """Write score directly onto the brain so sortPopulation can read it."""
        self.brain.score += score


    # --- End conditions ----------------------------------------------------

    def _is_over(self) -> bool :
        if self.game.game_over :
            return True
        if self.turns >= self.max_turns :
            return True
        if self.brain.score < self.lowest_score_allowed :
            return True
        return False

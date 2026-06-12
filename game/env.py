import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .constants import FIELDSIZE, Direction, FieldType, Cell
from .logic import GameLogic


class BombermanSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, level: int = 1, render_mode: str | None = None,
                 max_steps: int = 1000):
        super().__init__()
        self.level = level
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0, high=len(Cell) - 1,
            shape=(FIELDSIZE, FIELDSIZE),
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(4)

        self._game = GameLogic(level)
        self._step_count = 0
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium-Interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)
        self._game = GameLogic(self.level)
        self._step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        ate_food, died = self._game.step(int(action))
        self._step_count += 1

        if ate_food:
            reward = 1.0
        elif died:
            reward = -1.0
        else:
            reward = 0.0

        terminated = died
        truncated = (not died) and (self._step_count >= self.max_steps)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        grid = np.zeros((FIELDSIZE, FIELDSIZE), dtype=np.int8)

        # Terrain: FREE bleibt 0, WALL und EXPLODED direkt übernehmen
        for y in range(FIELDSIZE):
            for x in range(FIELDSIZE):
                ft = self._game.grid[y][x]
                if ft == FieldType.WALL:
                    grid[y][x] = Cell.WALL
                elif ft == FieldType.EXPLODED:
                    grid[y][x] = Cell.EXPLODED

        # Bombe
        if self._game.bomb and self._game.bomb_pos is not None:
            bx, by = self._game.bomb_pos
            grid[by][bx] = Cell.BOMB

        # Essen (kann von Explosion überdeckt sein → Explosion hat Vorrang)
        fx, fy = self._game.food_pos
        if grid[fy][fx] == Cell.FREE:
            grid[fy][fx] = Cell.FOOD

        # Schlangenkörper
        for x, y in self._game.snake[1:]:
            grid[y][x] = Cell.SNAKE_BODY

        # Schlangenkopf (zuletzt, hat höchste Priorität)
        if self._game.snake:
            hx, hy = self._game.snake[0]
            grid[hy][hx] = Cell.SNAKE_HEAD

        return grid

    def _get_info(self) -> dict:
        return {
            "score": self._game.score,
            "snake_length": len(self._game.snake),
            "step": self._step_count,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self) -> None:
        from .renderer import Renderer
        if self._renderer is None:
            self._renderer = Renderer()
        self._renderer.draw(self._game)

    def _get_rgb_array(self) -> np.ndarray:
        from .renderer import Renderer
        if self._renderer is None:
            self._renderer = Renderer()
        return self._renderer.get_rgb_array(self._game)

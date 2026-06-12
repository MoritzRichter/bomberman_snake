import numpy as np
from game.env import BombermanSnakeEnv
from game.constants import FIELDSIZE, Cell


class NeuralNetwork:
    """Einfaches Feedforward-Netz: 100 -> 64 (tanh) -> 4 (linear)."""

    # Architektur
    INPUT_SIZE  = FIELDSIZE * FIELDSIZE   # 100
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 4                        # LEFT RIGHT UP DOWN

    GENOME_SIZE = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE
                   + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE)  # 6724

    def __init__(self, genome: np.ndarray):
        assert len(genome) == self.GENOME_SIZE
        idx = 0

        s = self.INPUT_SIZE * self.HIDDEN_SIZE
        self.W1 = genome[idx:idx + s].reshape(self.INPUT_SIZE, self.HIDDEN_SIZE)
        idx += s

        self.b1 = genome[idx:idx + self.HIDDEN_SIZE]
        idx += self.HIDDEN_SIZE

        s = self.HIDDEN_SIZE * self.OUTPUT_SIZE
        self.W2 = genome[idx:idx + s].reshape(self.HIDDEN_SIZE, self.OUTPUT_SIZE)
        idx += s

        self.b2 = genome[idx:idx + self.OUTPUT_SIZE]

    @classmethod
    def random(cls) -> "NeuralNetwork":
        genome = np.random.randn(cls.GENOME_SIZE) * 0.1
        return cls(genome)

    def predict(self, obs: np.ndarray) -> int:
        """Observation (10×10) → Aktion (0-3)."""
        x = obs.flatten().astype(np.float32) / (len(Cell) - 1)   # normalisiert 0..1
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return int(np.argmax(out))


# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """Genetischer Algorithmus zur Optimierung der Netzgewichte."""

    def __init__(
        self,
        pop_size: int = 100,
        level: int = 1,
        max_steps: int = 500,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.05,
        tournament_k: int = 5,
        elite_count: int = 2,
    ):
        self.pop_size      = pop_size
        self.level         = level
        self.max_steps     = max_steps
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_k  = tournament_k
        self.elite_count   = elite_count

        # Population: Array (pop_size, GENOME_SIZE)
        self.population = np.array([
            np.random.randn(NeuralNetwork.GENOME_SIZE) * 0.1
            for _ in range(pop_size)
        ])
        self.fitnesses = np.zeros(pop_size)

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def evaluate(self, genome: np.ndarray) -> float:
        """Eine vollständige Episode durchlaufen, Gesamtreward zurückgeben."""
        nn = NeuralNetwork(genome)
        env = BombermanSnakeEnv(level=self.level, max_steps=self.max_steps)
        obs, _ = env.reset()

        total_reward = 0.0
        while True:
            action = nn.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        env.close()
        return total_reward

    def evaluate_all(self) -> None:
        """Fitness aller Individuen berechnen."""
        for i, genome in enumerate(self.population):
            self.fitnesses[i] = self.evaluate(genome)

    # ------------------------------------------------------------------
    # Selektion, Crossover, Mutation
    # ------------------------------------------------------------------

    def tournament_select(self) -> np.ndarray:
        """k zufällige Individuen ziehen, bestes Genom zurückgeben."""
        indices = np.random.choice(self.pop_size, self.tournament_k, replace=False)
        best = indices[np.argmax(self.fitnesses[indices])]
        return self.population[best].copy()

    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Uniform Crossover: jedes Gen mit 50 % Wahrscheinlichkeit von a oder b."""
        mask = np.random.rand(len(a)) < 0.5
        return np.where(mask, a, b)

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Gaußsches Rauschen auf zufällig gewählte Gene addieren."""
        mask  = np.random.rand(len(genome)) < self.mutation_rate
        noise = np.random.randn(len(genome)) * self.mutation_sigma
        return genome + mask * noise

    # ------------------------------------------------------------------
    # Eine Generation
    # ------------------------------------------------------------------

    def evolve_one_generation(self) -> dict:
        """Fitness berechnen und eine neue Generation erzeugen.

        Rückgabe: {'best': float, 'mean': float, 'worst': float}
        """
        self.evaluate_all()

        sorted_idx = np.argsort(self.fitnesses)[::-1]   # absteigend
        new_pop = []

        # Elitismus: beste Individuen unverändert übernehmen
        for i in range(self.elite_count):
            new_pop.append(self.population[sorted_idx[i]].copy())

        # Rest: Selektion → Crossover → Mutation
        while len(new_pop) < self.pop_size:
            parent_a = self.tournament_select()
            parent_b = self.tournament_select()
            child    = self.crossover(parent_a, parent_b)
            child    = self.mutate(child)
            new_pop.append(child)

        self.population = np.array(new_pop)

        return {
            "best":  float(self.fitnesses[sorted_idx[0]]),
            "mean":  float(self.fitnesses.mean()),
            "worst": float(self.fitnesses[sorted_idx[-1]]),
        }

    # ------------------------------------------------------------------
    # Trainingsloop
    # ------------------------------------------------------------------

    def run(self, generations: int, callback=None) -> None:
        """Mehrere Generationen trainieren.

        callback(gen, stats) wird nach jeder Generation aufgerufen (optional).
        """
        for gen in range(generations):
            stats = self.evolve_one_generation()
            if callback:
                callback(gen, stats)

    # ------------------------------------------------------------------
    # Modell speichern / laden
    # ------------------------------------------------------------------

    def best_genome(self) -> np.ndarray:
        """Bestes Genom der aktuellen Population zurückgeben."""
        return self.population[np.argmax(self.fitnesses)].copy()

    def save(self, path: str) -> None:
        """Bestes Genom als .npy speichern."""
        np.save(path, self.best_genome())

    def load(self, path: str) -> None:
        """Gespeichertes Genom laden und als erstes Individuum einsetzen."""
        genome = np.load(path)
        assert len(genome) == NeuralNetwork.GENOME_SIZE
        self.population[0] = genome

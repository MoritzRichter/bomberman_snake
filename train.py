import time
import argparse
import numpy as np
from ml.genetic import GeneticAlgorithm, NeuralNetwork
from game.env import BombermanSnakeEnv


# ── Hyperparameter ──────────────────────────────────────────────────────────
DEFAULTS = dict(
    generations    = 500,
    pop_size       = 100,
    level          = 1,
    max_steps      = 500,
    mutation_rate  = 0.10,
    mutation_sigma = 0.05,
    tournament_k   = 5,
    elite_count    = 2,
    save_path      = "best_genome.npy",
    log_interval   = 10,    # alle N Generationen eine Zeile ausgeben
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bomberman Snake – Training")
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=type(val),
                       default=val, help=f"(default: {val})")
    return p.parse_args()


def format_bar(value: float, best_ever: float, width: int = 20) -> str:
    """Einfacher ASCII-Fortschrittsbalken relativ zum bisher besten Score."""
    if best_ever <= 0:
        filled = 0
    else:
        filled = max(0, int(width * min(value / best_ever, 1.0)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def play_episode(genome: np.ndarray, level: int, max_steps: int) -> dict:
    """Eine Episode mit dem besten Genom durchlaufen, Details zurückgeben."""
    nn  = NeuralNetwork(genome)
    env = BombermanSnakeEnv(level=level, max_steps=max_steps)
    obs, _ = env.reset()

    steps = 0
    total_reward = 0.0
    while True:
        action = nn.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return {"score": info["score"], "steps": steps, "reward": total_reward}


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Bomberman Snake – Genetischer Algorithmus")
    print("=" * 60)
    print(f"  Level          : {args.level}")
    print(f"  Generationen   : {args.generations}")
    print(f"  Population     : {args.pop_size}")
    print(f"  Max-Schritte   : {args.max_steps}")
    print(f"  Mutationsrate  : {args.mutation_rate}")
    print(f"  Mutations-Sigma: {args.mutation_sigma}")
    print(f"  Turnier-k      : {args.tournament_k}")
    print(f"  Eliten         : {args.elite_count}")
    print(f"  Speicherpfad   : {args.save_path}")
    print("=" * 60)
    print(f"  {'Gen':>5}  {'Best':>7}  {'Mean':>7}  {'Worst':>7}  {'BestEver':>9}  Zeit/Gen")
    print("-" * 60)

    ga = GeneticAlgorithm(
        pop_size       = args.pop_size,
        level          = args.level,
        max_steps      = args.max_steps,
        mutation_rate  = args.mutation_rate,
        mutation_sigma = args.mutation_sigma,
        tournament_k   = args.tournament_k,
        elite_count    = args.elite_count,
    )

    best_ever        = -float("inf")
    best_ever_genome = None
    t_start          = time.time()

    for gen in range(args.generations):
        t0    = time.time()
        stats = ga.evolve_one_generation()
        dt    = time.time() - t0

        if stats["best"] > best_ever:
            best_ever        = stats["best"]
            best_ever_genome = ga.best_genome()
            np.save(args.save_path, best_ever_genome)

        if gen % args.log_interval == 0 or gen == args.generations - 1:
            bar = format_bar(stats["best"], max(best_ever, 1))
            print(
                f"  {gen:5d}  {stats['best']:7.2f}  {stats['mean']:7.2f}"
                f"  {stats['worst']:7.2f}  {best_ever:9.2f}  {dt:.1f}s  {bar}"
            )

    total = time.time() - t_start
    print("-" * 60)
    print(f"  Training abgeschlossen in {total:.0f}s")
    print(f"  Bester Score (Fitness): {best_ever:.2f}")
    print(f"  Bestes Genom gespeichert: {args.save_path}")

    # ── Abschluss-Episode mit bestem Genom ──────────────────────────────────
    if best_ever_genome is not None:
        result = play_episode(best_ever_genome, args.level, args.max_steps)
        print(f"  Kontroll-Episode: Score={result['score']}  "
              f"Schritte={result['steps']}  Reward={result['reward']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

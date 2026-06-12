import sys
import pygame
from game.env import BombermanSnakeEnv
from game.constants import Direction

FPS        = 5      # Spielgeschwindigkeit (Schritte pro Sekunde)
LEVEL      = 1      # Starklevel (1-3)
MAX_STEPS  = 2000


def main() -> None:
    env   = BombermanSnakeEnv(level=LEVEL, render_mode="human", max_steps=MAX_STEPS)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    action = Direction.RIGHT   # Startrichtung

    print("Bomberman Snake – Steuerung:")
    print("  Pfeiltasten : Richtung aendern")
    print("  1 / 2 / 3   : Level wechseln")
    print("  R           : Neustart")
    print("  ESC / Q     : Beenden")

    running = True
    while running:
        # ---- Events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_LEFT:
                    action = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    action = Direction.UP
                elif event.key == pygame.K_DOWN:
                    action = Direction.DOWN
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    action = Direction.RIGHT
                elif event.key == pygame.K_1:
                    env = BombermanSnakeEnv(level=1, render_mode="human",
                                            max_steps=MAX_STEPS)
                    obs, info = env.reset()
                    action = Direction.RIGHT
                elif event.key == pygame.K_2:
                    env = BombermanSnakeEnv(level=2, render_mode="human",
                                            max_steps=MAX_STEPS)
                    obs, info = env.reset()
                    action = Direction.RIGHT
                elif event.key == pygame.K_3:
                    env = BombermanSnakeEnv(level=3, render_mode="human",
                                            max_steps=MAX_STEPS)
                    obs, info = env.reset()
                    action = Direction.RIGHT

        # ---- Schritt ----
        obs, reward, terminated, truncated, info = env.step(action)

        pygame.display.set_caption(
            f"Bomberman Snake  |  Score: {info['score']}  "
            f"Laenge: {info['snake_length']}  Schritt: {info['step']}"
        )

        # ---- Game Over / Truncated ----
        if terminated:
            print(f"Game Over!  Score: {info['score']}  (R = Neustart)")
            pygame.time.wait(800)
            obs, info = env.reset()
            action = Direction.RIGHT

        if truncated:
            print(f"Zeit abgelaufen.  Score: {info['score']}  (R = Neustart)")
            pygame.time.wait(800)
            obs, info = env.reset()
            action = Direction.RIGHT

        clock.tick(FPS)

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

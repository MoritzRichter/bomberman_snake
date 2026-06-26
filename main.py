import sys
import pygame
from game.logic import GameLogic
from game.renderer import Renderer
from game.constants import Direction

FPS       = 5
MAX_STEPS = 2000


def main() -> None:
    cur_level = 1
    game      = GameLogic(level=cur_level)
    renderer  = Renderer()
    clock     = pygame.time.Clock()
    action    = Direction.RIGHT
    step      = 0

    print("Bomberman Snake – Steuerung:")
    print("  Pfeiltasten : Richtung aendern")
    print("  1 / 2 / 3   : Level wechseln")
    print("  R           : Neustart")
    print("  ESC / Q     : Beenden")

    running = True
    while running:
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
                    game.reset()
                    action = Direction.RIGHT
                    step = 0
                elif event.key == pygame.K_1:
                    cur_level = 1
                    game = GameLogic(level=cur_level)
                    action = Direction.RIGHT
                    step = 0
                elif event.key == pygame.K_2:
                    cur_level = 2
                    game = GameLogic(level=cur_level)
                    action = Direction.RIGHT
                    step = 0
                elif event.key == pygame.K_3:
                    cur_level = 3
                    game = GameLogic(level=cur_level)
                    action = Direction.RIGHT
                    step = 0

        ate_food, died = game.step(int(action))
        step += 1
        renderer.draw(game)

        pygame.display.set_caption(
            f"Bomberman Snake  |  Score: {game.score}  "
            f"Laenge: {len(game.snake)}  Schritt: {step}"
        )

        if died:
            print(f"Game Over!  Score: {game.score}  (R = Neustart)")
            pygame.time.wait(800)
            game.reset()
            action = Direction.RIGHT
            step = 0
        elif step >= MAX_STEPS:
            print(f"Zeit abgelaufen.  Score: {game.score}  (R = Neustart)")
            pygame.time.wait(800)
            game.reset()
            action = Direction.RIGHT
            step = 0

        clock.tick(FPS)

    renderer.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

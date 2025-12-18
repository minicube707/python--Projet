import pygame
import numpy as np
from numba import njit

WIDTH = 800
ROWS = 100

WHITE = (255,255,255)
LIGHT = (200,200,200)
MID   = (130,130,130)
DARK  = (60,60,60)
BLACK = (0,0,0)

def draw(win, grid):
    win.fill(BLACK)
    rows = grid.shape[0]
    gap = WIDTH // rows

    for i in range(rows):
        for j in range(rows):
            v = grid[i, j]
            if v == 0: color = WHITE
            elif v == 1: color = LIGHT
            elif v == 2: color = MID
            else: color = DARK

            pygame.draw.rect(
                win, color,
                (j*gap, i*gap, gap, gap)
            )

    pygame.display.update()


@njit
def relax_sandpile(grid):
    rows, cols = grid.shape
    unstable = True

    while unstable:
        unstable = False

        for i in range(rows):
            for j in range(cols):
                if grid[i, j] >= 4:
                    grid[i, j] -= 4
                    unstable = True

                    if i > 0:
                        grid[i - 1, j] += 1
                    if i < rows - 1:
                        grid[i + 1, j] += 1
                    if j > 0:
                        grid[i, j - 1] += 1
                    if j < cols - 1:
                        grid[i, j + 1] += 1


def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption("SOC Sandpile (Numba)")

    grid = np.random.random((ROWS, ROWS)) * 5
    grid = np.floor(grid)

    clock = pygame.time.Clock()
    run = True

    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # ajout d'un grain
        i = np.random.randint(0, ROWS)
        j = np.random.randint(0, ROWS)
        grid[i, j] += 1

        # relaxation SOC (rapide)
        relax_sandpile(grid)

        # affichage
        draw(win, grid)

    pygame.quit()


if __name__ == "__main__":
    main()

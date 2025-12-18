import numpy as np
import pygame
import matplotlib.pyplot as plt
import time
from collections import deque

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Critical State Self Organizing")

BLACK =         (0, 0, 0)
DARK_GREY =    (64, 64, 64)
MID_GREY =    (128, 128, 128)
LIGHT_GREY =    (196, 196, 196)
WHITE =         (255, 255, 255)

# paramètres
MIN_EVENTS = 1     # éviter les erreurs au début

def draw_grid (win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))

        for j in range(rows):
            pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))


def draw(win, rows, width, grid):
    win.fill(BLACK)
    gap = width // rows

    for row in range(rows):
        for col in range(rows):
            value = grid[row, col]
            if value == 0:
                color = WHITE
            elif value == 1:
                color = LIGHT_GREY
            elif value == 2:
                color = MID_GREY
            elif value == 3:
                color = DARK_GREY
            else:
                color = BLACK
            pygame.draw.rect(win, color, (col * gap, row * gap, gap, gap))

    draw_grid(win, rows, width)
    pygame.display.update()

    
def relax_and_count(win, rows, width, grid):
    avalanche_size = 0

    while True:
        unstable = grid >= 4
        n_topplings = np.sum(unstable)

        if n_topplings == 0:
            break

        avalanche_size += n_topplings

        grid[unstable] -= 4

        # diffusion vers les voisins
        grid[1:, :]  += unstable[:-1, :]
        grid[:-1, :] += unstable[1:, :]
        grid[:, 1:]  += unstable[:, :-1]
        grid[:, :-1] += unstable[:, 1:]
        draw(win, rows, width, grid)

    return avalanche_size
        

#Main algorithm
def main (win , width):

    rows = 100

    #Gen grid
    grid = np.random.random((rows, rows)) * 5
    grid = np.floor(grid)
    relax_and_count(win, rows, width, grid)
    avalanche_sizes = []

    run = True
    stop = False
    while run:

        #Pygame event
        for event in pygame.event.get():

            #Quit pygame
            if event.type == pygame.QUIT:
                return (False, grid)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                     return (False, grid)
                if event.key == pygame.K_SPACE:
                        stop = not stop

        if (not stop):   

            draw(win, rows, width, grid)

            randon_x = np.random.randint(0, rows)
            randon_y = np.random.randint(0, rows)
            grid[randon_y, randon_x] += 1

            s = relax_and_count(win, rows, width, grid)
            if s > 0:
                avalanche_sizes.append(s)

            # graphique OCCASIONNEL
            if len(avalanche_sizes) > MIN_EVENTS:
                sizes = np.array(avalanche_sizes)
                bins = np.logspace(
                    np.log10(sizes.min()),
                    np.log10(sizes.max()),
                    40
                )

                hist, edges = np.histogram(sizes, bins=bins, density=True)
                bin_centers = np.sqrt(edges[:-1] * edges[1:])
                mask = hist > 0

                x = np.log10(bin_centers[mask])
                y = np.log10(hist[mask])

                slope, intercept = np.polyfit(x, y, 1)

                x_fit = np.logspace(np.log10(bin_centers.min()),
                                    np.log10(bin_centers.max()), 100)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))

                plt.clf()
                plt.loglog(bin_centers, hist, 'o')
                plt.loglog(x_fit, y_fit, '-', label=f"Pente = {slope:.2f}")
                plt.xlabel("Taille de l'avalanche")
                plt.ylabel("P(s)")
                plt.legend()
                plt.pause(0.001)

main(WIN, WIDTH)
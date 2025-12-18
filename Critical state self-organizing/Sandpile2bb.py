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

    
def check_state(win, width, grid, rows):

    while np.any(grid >= 4):
        
        unstable = grid >= 4
        grid[unstable] -= 4

        # Ajouter aux voisins avec conditions périodiques
        grid += np.roll(unstable, 1, axis=0)    # voisin du haut
        grid += np.roll(unstable, -1, axis=0)   # voisin du bas
        grid += np.roll(unstable, 1, axis=1)    # voisin de gauche
        grid += np.roll(unstable, -1, axis=1)   # voisin de droite

        draw(win, rows, width, grid)

    return grid
        

#Main algorithm
def main (win , width):

    rows = 100

    #Gen grid
    grid = np.random.random((rows, rows)) * 5
    grid = np.floor(grid)

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
            grid = check_state(win, width, grid, rows)


main(WIN, WIDTH)
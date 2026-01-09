import pygame
import random
import numpy as np
import time
import os

pygame.init()

# Crée une fenêtre de 800x600 pixels
WIDTH, HEIGHT = 800, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Définit le titre de la fenêtre
pygame.display.set_caption("Basic pygame")


# Définit la couleur blanche
WHITE = [255, 255, 255]
GREY =  [127, 127, 127]
BLACK = [0, 0, 0]

RED =   [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]

CYAN = [0, 255, 255]
YELLOW = [255, 255, 0]
PURPLE = [255, 0, 255]

ORANGE = [255, 165, 0]

SHADOW = [50, 50, 50]


#Import the music
module_dir = os.path.dirname(__file__)
os.chdir(module_dir)
music = pygame.mixer.music.load("Tetris_song.mp3")

#Return true if a figure is in pos 0 height
def end_game(all_figure):
    end = False
    for figure in all_figure:
        for pos in figure:

            if pos[1]==0:
                end = True

    return end


#Delete the line
def delete_line(all_figure,  list_colour, list_y_line, grid):

    #For every line in game   
    for index_line, y_line in enumerate (list_y_line):

        #Reset the grid
        grid[y_line,:] = 0
        x = np.arange(10)
        y = np.ones((10), np.int8) * y_line
        line_pos = np.dstack((x, y))[0]     #The tab with the coordinate of the line

        #For every cell in the grid
        #Update the grid
        for row in range(y_line-1, -1, -1):
            for col in range(cols):
                        
                #If there is a bloc
                if grid[row, col] == 1: 
                    
                    #Go down one step
                    grid[row +1 , col] = 1
                    grid[row, col] = 0
                 

        new_all_figure = []
        new_list_colour = []

        #For every figure 
        for index_figure, figure in enumerate (all_figure):

            len_figure = len(figure)
            new_figure = np.array([], np.int16)

            #For every bloc
            for bloc in figure:
                add = True
                    
                #For every position of the line
                for pos in line_pos:
                        
                    #If the bloc is on the line don't save it
                    if np.array_equal(bloc, pos):
                        len_figure -= 1
                        add = False

                #If we save the bloc
                if add:

                    #If the bloc is above the line, go down one step
                    if bloc[1] < y_line:
                        new_bloc = [bloc[0], bloc[1]+1]
                    #Else
                    else:
                        new_bloc = bloc
                    new_figure = np.append(new_figure, new_bloc)

            #If the figure is delete, delete also the colour associate
            if len_figure > 0:
                new_list_colour.append(list_colour[index_figure])

            #If figure isn't empty, save it
            if not np.array_equal(new_figure, np.array([])):
                new_figure = new_figure.reshape((-1, 2))
                new_all_figure.append(new_figure)

        #If there is another, restart the variable
        if len(list_y_line) > index_line:
            all_figure = new_all_figure
            list_colour = new_list_colour

    return new_all_figure, new_list_colour, grid


#Verify if there a line
def verify_line(rows, all_figure, list_colour, grid):

    #verify if there is a line in game
    list_y_line = []
    for i in range(rows):

        if all(grid[i,:] == 1):
            list_y_line.append(i)

    #If there is a line in game
    if len(list_y_line) > 0:
        new_all_figure, new_list_colour, grid = delete_line(all_figure,  list_colour, list_y_line, grid)
        
    #If there is no line, do nothing
    else : 
        new_all_figure = all_figure
        new_list_colour = list_colour

    nb_line = len(list_y_line)

    return new_all_figure, new_list_colour, nb_line, grid


def update_grid(all_figure):
    
    grid = np.zeros((rows, cols), np.int8)

    for figure in all_figure:
        for bloc in figure:
            grid[bloc[1], bloc[0]] = 1
        
    return grid

#Turn the figure
def turn_figure(figure, num, state):
    
    max_y = np.max(figure[:, 1], 0)
    x = int(np.mean(figure[:, 0], 0))


    #Figure1    Tétrimino I
    if num == 1:
        if state == 0:
            # X
            # X
            # X
            # X
            figure = np.array([[x, max_y-2], [x, max_y-1], [x, max_y], [x, max_y+1]])
            new_state = 1
        
        else:
            # X X X X
            figure = np.array([[x-1, max_y], [x, max_y], [x+1, max_y], [x+2, max_y]])
            new_state = 0
            

    #Figure2    Tétrimino O
    elif num == 2:
        # X X
        # X X
        figure = figure
        new_state = 0

    #Figure3    Tétrimino T
    elif num == 3:
        if state == 0:
            # X 
            # X X
            # X
            figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x+1, max_y]])
            new_state = 1
      

        elif state == 1:
            #   X
            # X X X
            figure = np.array([[x+1, max_y -1], [x, max_y -1], [x-1, max_y -1], [x, max_y-2]])
            new_state = 2
   

        elif state == 2:
            #   X
            # X X
            #   X
            figure = np.array([[x, max_y-1], [x, max_y], [x, max_y + 1], [x-1, max_y]])
            new_state = 3

        
        else:
            # X X X
            #   X
            figure = np.array([[x+2, max_y], [x+1, max_y], [x, max_y], [x+1, max_y+1]])
            new_state = 0
 

    #Figure4    Tétrimino L
    elif num == 4:
        if state == 0:
            # X 
            # x
            # x X
            figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x+1, max_y+1]])
            new_state = 1

        
        elif state == 1:
            #     X
            # X X X
            figure = np.array([[x, max_y-1], [x+1, max_y-1], [x+2, max_y-1], [x+2, max_y-2]])
            new_state = 2

        elif state == 2:
            # X X
            #   X
            #   X
            figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x-1, max_y-1]])
            new_state = 3

        else:
            # X X X
            # X
            figure = np.array([[x, max_y], [x+1, max_y], [x+2, max_y], [x, max_y+1]])
            new_state = 0
    
    #Figure5    Tétrimino J
    elif num == 5:
        if state == 0:
            # X X
            # X
            # X
            figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x+1, max_y-1]])
            new_state = 1

        
        elif state == 1: 
            # X
            # X X X           
            figure = np.array([[x-1, max_y-1], [x, max_y-1], [x+1, max_y-1], [x-1, max_y-2]])
            new_state = 2

        elif state == 2:
            #   X
            #   X 
            # X X
            figure = np.array([[x+1, max_y-1], [x+1, max_y], [x+1, max_y+1], [x, max_y+1]])
            new_state = 3

        else:
            # X X X
            #     X
            figure = np.array([[x, max_y], [x+1, max_y], [x+2, max_y], [x+2, max_y+1]])
            new_state = 0

    #Figure6    Tétrimino Z
    elif num == 6:
        if state == 0:
            #   X
            # X X
            # X
            figure = np.array([[x-1, max_y+1], [x-1, max_y], [x, max_y], [x, max_y-1]])
            new_state = 1
        
        else:
            #  X X
            #    X X
            figure = np.array([[x, max_y], [x+1, max_y], [x+1, max_y+1], [x+2, max_y+1]])
            new_state = 0

    #Figure7   Tétrimino S
    elif num == 7:
        if state == 0:
            # X
            # X X
            #   X
            figure = np.array([[x, max_y-1], [x, max_y], [x+1, max_y], [x+1, max_y+1]])
            new_state = 1
        
        else:
            #   X X
            # X X
            figure = np.array([[x-1, max_y+1], [x, max_y+1], [x, max_y], [x+1, max_y]])
            new_state = 0
        
    return figure, new_state


#Return the dimension of the grid 
def dim_grid(WIDTH, HEIGHT, rows, cols, marge):

    # Calcul des dimensions de chaque cellule
    cell_width = WIDTH // (cols + marge)  
    cell_height = HEIGHT // (rows + marge)
    
    # Utiliser le plus petit des deux pour garder des cellules carrées
    cell_size = min(cell_width, cell_height)
    
    # Calcul du décalage pour centrer la grille
    x_offset = (WIDTH - cell_size * cols) // 2
    y_offset = (HEIGHT - cell_size * rows) // 2

    return x_offset, y_offset, cell_size


#Function with all the fonction draw
def draw(WIN, HEIGHT, rows, cols, cell_size, x_offset, y_offset, all_figure, list_colour, colour, figure, next_num):

    marge_square = 0.1

    #Draw function
    draw_grid(WIN, rows, cols, cell_size, x_offset, y_offset)
    draw_figure(WIN, x_offset, y_offset, cell_size, colour, figure, marge_square)
    draw_next_figure(x_offset, HEIGHT, cell_size, next_num, marge_square)
    draw_land_figure(WIN, x_offset, y_offset, cell_size, all_figure, list_colour, marge_square)
    

    #Text
    FONT = pygame.font.SysFont("trebuchetms", 20)
    text_score = FONT.render("Score: " + str(score), 1, WHITE)
    text_level = FONT.render("Level: " + str(level), 1, WHITE)
    WIN.blit(text_score, (x_offset//2 - text_score.get_width()//2, 3*HEIGHT//4))
    WIN.blit(text_level, (x_offset//2 - text_score.get_width()//2, 3*HEIGHT//4 - text_score.get_height()))


#Draw the grid
def draw_grid(win, rows, cols, cell_size, x_offset, y_offset):

    # Dessiner la grille
    for row in range(rows):
        for col in range(cols):
            x = x_offset + col * cell_size
            y = y_offset + row * cell_size
            pygame.draw.rect(win, WHITE, (x, y, cell_size, cell_size), 1)


#Draw the next figure         
def draw_next_figure(x_offset, HEIGHT, cell_size, next_num, marge):

    #Grid   
    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 1*cell_size, HEIGHT//2, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 0*cell_size, HEIGHT//2, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 1*cell_size, HEIGHT//2, cell_size, cell_size), 1)

    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size), 1)

    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 2*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 2*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2 - 2*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 2*cell_size, cell_size, cell_size), 1)

    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 1*cell_size, HEIGHT//2 + 1*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 + 0*cell_size, HEIGHT//2 + 1*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2 + 1*cell_size, cell_size, cell_size), 1)
    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 1*cell_size, HEIGHT//2 + 1*cell_size, cell_size, cell_size), 1)

    #Figure
    if next_num == 1:
        pygame.draw.rect(WIN, CYAN, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, CYAN, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, CYAN, (x_offset//2 - 2*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, CYAN, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [1, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-2, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -1], cell_size, marge)
    
    elif next_num == 2:
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -1], cell_size, marge)

    elif next_num == 3:
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 - 0*cell_size, HEIGHT//2 + 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 1], cell_size, marge)


    elif next_num == 4:
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [1, -1], cell_size, marge)

    elif next_num == 5:
        pygame.draw.rect(WIN, BLUE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -1], cell_size, marge)

    elif next_num == 6:
        pygame.draw.rect(WIN, RED, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -1], cell_size, marge)

    else:
        pygame.draw.rect(WIN, GREEN, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, GREEN, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, GREEN, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, GREEN, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [1, -1], cell_size, marge)


    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2 - 2*cell_size, 4*cell_size, 4*cell_size), 4)


#Draw the figure land
def draw_land_figure(WIN, x_offset, y_offset, gap, all_figure, list_colour, marge):

    for index, figure in enumerate (all_figure):
        for pos in figure:
            pygame.draw.rect(WIN, list_colour[index], (x_offset + pos[0]*gap, y_offset + pos[1]*gap, gap, gap))
            draw_shaders(x_offset, y_offset, pos, gap, marge)


#Draw the figure who fall
def draw_figure(WIN, x_offset, y_offset, gap, colour, figure, marge):

    for pos in figure:
        pygame.draw.rect(WIN, colour, (x_offset + pos[0]*gap, y_offset + pos[1]*gap, gap, gap))
        draw_shaders(x_offset, y_offset, pos, gap, marge)


#Draw the shaders
def draw_shaders(x_offset, y_offset, pos, gap, marge):
            
    #The shaders
    white_poly = ((x_offset + pos[0]*gap, y_offset + pos[1]*gap), (x_offset + (pos[0] + 1)*gap, y_offset + pos[1]*gap), (x_offset + (pos[0] + (1 - marge))*gap, y_offset + (pos[1] + marge)*gap), (x_offset + (pos[0] + marge)*gap, y_offset + (pos[1] + marge)*gap))
    pygame.draw.polygon(WIN, WHITE, white_poly)

    dark_poly = ((x_offset + (pos[0])*gap, y_offset + (pos[1] + 1)*gap), (x_offset + (pos[0] + 1)*gap, y_offset + (pos[1] + 1)*gap), (x_offset + (pos[0] + (1 -marge))*gap, y_offset + (pos[1] + (1 - marge))*gap), (x_offset + (pos[0] + marge)*gap, y_offset + (pos[1] + (1 - marge))*gap))
    pygame.draw.polygon(WIN, SHADOW, dark_poly)

    left_poly = ((x_offset + (pos[0])*gap, y_offset + (pos[1])*gap), (x_offset + (pos[0])*gap, y_offset + (pos[1] +1 )*gap), (x_offset + (pos[0] + marge)*gap, y_offset + (pos[1] + (1 - marge))*gap), (x_offset + (pos[0] + marge)*gap, y_offset + (pos[1] + marge)*gap))
    pygame.draw.polygon(WIN, GREY, left_poly)

    right_poly = ((x_offset + (pos[0]+1)*gap, y_offset + (pos[1])*gap), (x_offset + (pos[0] + 1)*gap, y_offset + (pos[1] + 1)*gap), (x_offset + (pos[0] + (1 - marge))*gap, y_offset + (pos[1] + (1 - marge))*gap), (x_offset + (pos[0] + (1 - marge))*gap, y_offset + (pos[1] + marge)*gap))
    pygame.draw.polygon(WIN, GREY, right_poly)


#Return the new figure 
def add_figure(list_num):

    if len(list_num) == 0:
        list_num = [1, 2, 3, 4, 5, 6, 7]

    rand = random.randint(0, len(list_num)-1)
    num = list_num[rand]      
    list_num.remove(num) 

    #Figure1    Tétrimino I
    if num == 1:
        colour = CYAN
        figure = np.array([[3, 0], [4, 0], [5, 0], [6, 0]])
        nb_state = 2

    #Figure2    Tétrimino O
    elif num == 2:
        colour = YELLOW
        figure = np.array([[4, -1], [4, 0], [5, 0], [5, -1]])
        nb_state = 1

    #Figure3    Tétrimino T
    elif num == 3:
        colour = PURPLE
        figure = np.array([[3, -1], [4, -1], [5, -1], [4, 0]])
        nb_state = 4
        
    #Figure4    Tétrimino L
    elif num == 4:
        colour = ORANGE
        figure = np.array([[3, -1], [4, -1], [5, -1], [3, 0]])
        nb_state = 4
        
    #Figure5    Tétrimino J
    elif num == 5:
        colour = BLUE
        figure = np.array([[3, -1], [4, -1], [5, -1], [5, 0]])
        nb_state = 4
        
    #Figure6    Tétrimino Z
    elif num == 6:
        colour = RED
        figure = np.array([[3, -1], [4, -1], [4, 0], [5, 0]])
        nb_state = 2
    
    #Figure7   Tétrimino S
    elif num == 7:
        colour = GREEN
        figure = np.array([[3, 0], [4, 0], [4, -1], [5, -1]])
        nb_state = 2
    
    return figure, colour, num, list_num, nb_state


#Update the score
def update_score(score, level, nb_line, tt_nb_line):

    if nb_line == 1:
        score += 40*level

    elif nb_line == 2:
        score += 100*level
            
    elif nb_line == 3:
        score += 300*level

    elif nb_line == 4:
        score += 1200*level

    #Update the level, the level incrise all the 10 lines delete
    tt_nb_line += nb_line
    level = (tt_nb_line // 10) +1

    return score, level, tt_nb_line


def make_score(test_figure, test_grid, rows, cols):
    
    malus = 0

    #Add the figure
    for test_bloc in test_figure:
        test_grid[test_bloc[1], test_bloc[0]] = 1

    #verify if there is a line in game
    for i in range(rows):
            
        #If there is a line
        if all(test_grid[i,:] == 1):
            test_grid[i,:] = 0

            #For every cell in the grid
            for row in range(i-1, -1, -1):
                for col in range(cols):
                        
                    #If there is a bloc
                    if test_grid[row, col] == 1: 
                        
                        #Go down one step
                        test_grid[row +1 , col] = 1
                        test_grid[row, col] = 0

    #Malus if an empty cell is surrounded by fill cell
    for x in range(cols-3):
        for y in range(rows-3):
            if np.array_equal(test_grid[y:y+3, x:x+3], np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
                malus += (rows -  y+1)*2

    return test_grid, malus


def find_position(grid, rows, cols, next_figure, figure, next_num, num, nb_state, next_nb_state):
    

    coef_matrix = np.arange(1, rows+1)[::-1]
    coef_matrix = coef_matrix.reshape(rows, 1)
    intial_next_figure = next_figure.copy()
    best_score = float("inf")
    
    """"""""""""""
    #First figure
    """"""""""""""
    #For every state of the figure
    for state1 in range(nb_state):

        max_x1 = np.max(figure[:, 0], 0) 
        min_x1 = np.min(figure[:, 0], 0)
        
        width_figure1 = max_x1 - min_x1

        #For every position on the x axis
        for delta_x1 in range(cols - width_figure1):
            
            contact1 = False
            nb_contact = 0
            delta_y1 = 0


            #While the figure doesn't touch the ground or an other figure
            while not contact1:

                #For every bloc in figure
                for bloc in figure:

                    #First verify if the figure touch the ground
                    if (bloc[1] + delta_y1 == rows):
                        contact1 = True

                    #Then verify if the figure touch an other figure
                    elif bloc[1] + delta_y1 >= 0: #avoid than the index is negative

                        #If there a bloc below 
                        if (grid[bloc[1] + delta_y1, bloc[0] + delta_x1 - min_x1] == 1):
                            nb_contact += 1
                            contact1 = True   


                #If the figure doesn't touch the ground or an other figure, it go down
                if not contact1:
                    
                    #Update the position of the figure
                    test_figure1 = np.array([], np.int8)
                    for bloc in figure:
                        test_bloc1 = np.array([], np.int8)
                        test_bloc1 = np.append(test_bloc1, bloc[0] + delta_x1 - min_x1)
                        test_bloc1 = np.append(test_bloc1, bloc[1] + delta_y1)
                        test_figure1 = np.append(test_figure1, test_bloc1)

                    delta_y1 +=1
                    test_figure1 = test_figure1.reshape(4, 2)

 
            """"""""""""""
            #Seconde figure
            """"""""""""""
            next_figure = intial_next_figure
            for state2 in range(next_nb_state):

                max_x2 = np.max(next_figure[:, 0], 0) 
                min_x2 = np.min(next_figure[:, 0], 0)
                width_figure2 = max_x2 - min_x2

                #For every position on the x axis
                for delta_x2 in range(cols - width_figure2):
                
                    contact2 = False
                    delta_y2 = 0

                    test_grid = grid.copy()  
                    for test_bloc1 in test_figure1:
                        test_grid[test_bloc1[1], test_bloc1[0]] = 1

                    #While the figure doesn't touch the ground or an other figure
                    while not contact2:

                        #For every bloc in figure
                        for next_bloc in next_figure:

                            #First verify if the figure touch the ground
                            if (next_bloc[1] + delta_y2 == rows):
                                contact2 = True

                            #Then verify if the figure touch an other figure
                            elif next_bloc[1] + delta_y2 >= 0 : #avoid than the index is negative

                                #If there a bloc below 
                                if (test_grid[next_bloc[1] + delta_y2, next_bloc[0] + delta_x2 - min_x2] == 1):
                                    contact2 = True   

                        #If the figure doesn't touch the ground or an other figure, it go down
                        if not contact2:
                        
                            #Update the position of the figure
                            test_figure2 = np.array([], np.int8)
                            for next_bloc in next_figure:
                                test_bloc2 = np.array([], np.int8)
                                test_bloc2 = np.append(test_bloc2, next_bloc[0] + delta_x2 - min_x2)
                                test_bloc2 = np.append(test_bloc2, next_bloc[1] + delta_y2)
                                test_figure2 = np.append(test_figure2, test_bloc2)

                            delta_y2 +=1
                            test_figure2 = test_figure2.reshape(4, 2)

                    test_grid, malus = make_score(test_figure2, test_grid, rows, cols)
                    
                    score_matrix = coef_matrix.T.dot(test_grid)
                    score = np.sum(score_matrix)
                    score += malus


                    #If the score is lower, save the parametre
                    if score < best_score:
                        best_score = score
                        best_state = state1
                        best_delta_x = delta_x1
                        best_nb_contact = nb_contact

                    #If the score is the same but, the figure has more neighbors, save the parametre
                    elif score == best_score and best_nb_contact < nb_contact:
                        best_score = score
                        best_state = state1
                        best_delta_x = delta_x1
                        best_nb_contact = nb_contact

                next_figure, _ = turn_figure(next_figure, next_num, state2)
        figure, _ = turn_figure(figure, num, state1)
    


    return best_state, best_delta_x


def place_figure(best_state, best_delta_x, figure, state, num, start, delay):

    fall = 0
    is_fall = True      # figure is fall ?
    contact_y = False   # the figure colide on the y axis ?

    # Is the figure must fall
    if time.time() >= start + delay:
        start = time.time()
        fall = 1

    if state != best_state:
        figure, new_state = turn_figure(figure, num, state)
    else:
        new_state = state

    min_x = np.min(figure[:, 0], 0)
    max_y = np.max(figure[:, 1], 0)

    #Verefy the colision
    for land_figure in all_figure:
        for bloc_A in land_figure:
            for bloc in figure:
                    
                #If the bloc A and B are the same there is a colision on the y axis
                bloc_B = np.array([bloc[0], bloc[1] + fall])
                if np.array_equal(bloc_A, bloc_B):
                    is_fall = False
                    contact_y = True
                    

    delta_x = 0
    if min_x != best_delta_x:

        if min_x > best_delta_x:
            delta_x = -1

        else:
            delta_x = 1

    new_figure = np.array([], np.int8)
    for bloc in figure:
            
        new_bloc = np.array([], np.int8)
        new_bloc = np.append(new_bloc, bloc[0] + delta_x)

        if max_y + fall < rows and not contact_y:
                new_bloc = np.append(new_bloc, bloc[1]  + fall)
        else:
            new_bloc = np.append(new_bloc, bloc[1])

        new_figure = np.append(new_figure, new_bloc)

    #Reshape figure because it containt 4 bloc withs 2 coordonates        
    new_figure = new_figure.reshape(4, 2)

    #If the figure touch the ground stop it fall
    if max_y == rows-1:
        is_fall = False

    return new_figure, new_state, start, is_fall

#Bool
run = True          #The game run ?
is_fall = True      #The figure fall ?
end = False         #The game is over ?
new_figure = True   #Must add a new figure ?
pause = False       #The game is in pause ?
sound_mute = False  #The sound is mute

#List
list_num = [1, 2, 3, 4, 5, 6, 7]        #The list of the figure
list_colour = []                        #The list of all the figure in game
all_figure = []                         #All the figure in game

#Int
marge = 0               #The marge between the edge of the window
rows = 20              #How many rows
cols = 10               #How many cols
delay = 0.1  
start_press = 0         #The delay before update the figure, for don't be too fast
score = 0               #The score
level = 1               #The level
tt_nb_line = 0          #How many line is deleted

#Time
start_new_figure = time.time() - delay

#The  gap between the edge of window
x_offset, y_offset, cell_size = dim_grid(WIDTH, HEIGHT, rows, cols, marge)

#The figure and the next
figure, colour, num, list_num, nb_state = add_figure(list_num)
state =  0

next_figure, next_colour, next_num, list_num, next_nb_state = add_figure(list_num)
next_state = 0

# Boucle principale
clock = pygame.time.Clock()
start = time.time()

#Play the music
pygame.mixer.music.play(-1)

grid = np.zeros((rows, cols), np.int8)
best_state, best_delta_x = find_position(grid, rows, cols, next_figure, figure, next_num, num, nb_state, next_nb_state)

while run:
            
    #Pygame event
    for event in pygame.event.get():
            
        #Quit pygame
        if event.type == pygame.QUIT:
            run = False

        #If a key is press
        if event.type == pygame.KEYDOWN:

            #Quit pygame
            if event.key == pygame.K_ESCAPE:
                run = False

            #Pause
            if event.key ==  pygame.K_SPACE:
                if pause:
                    pause = not pause
                
                else:
                    pause = not pause
            
            #Mute the sound
            if event.key ==  pygame.K_m:
                if sound_mute:
                    pygame.mixer.music.unpause()
                    sound_mute = not sound_mute
                else:
                    pygame.mixer.music.pause()
                    sound_mute = not sound_mute

    clock.tick(60)
    WIN.fill(BLACK)

    grid = update_grid(all_figure)
    
    #If the game isn't end
    if not end:

        #If pause
        if pause:
            #Draw the game
            draw(WIN, HEIGHT, rows, cols, cell_size, x_offset, y_offset, all_figure, list_colour, colour, figure, next_num)
            
            #Text
            PAUSE = pygame.font.SysFont("trebuchetms", 50)
            text_pause = PAUSE.render("PAUSE", 1, WHITE)
            WIN.blit(text_pause, (WIDTH//2 - text_pause.get_width()//2, HEIGHT//2 - text_pause.get_height()//2))


        #If the figure didn't fall and the delay is pass
        elif not is_fall:
            
            #Add the figure to  the figure land
            all_figure.append(figure)
            list_colour.append(colour)

            grid = update_grid(all_figure)
            
            #Verify if there is a line, and update the game
            all_figure, list_colour, nb_line, grid = verify_line(rows, all_figure, list_colour, grid)

            #The next figure become this currently
            figure = next_figure
            colour = next_colour
            num = next_num
            state = next_state
            nb_state = next_nb_state

            score, level, tt_nb_line = update_score(score, level, nb_line, tt_nb_line)

            #Create the next figure
            next_figure, next_colour, next_num, list_num, next_nb_state = add_figure(list_num)

            best_state, best_delta_x = find_position(grid, rows, cols, next_figure, figure, next_num, num, nb_state, next_nb_state)

            #Reset the time for delay of keys
            start = time.time()
            new_figure = True
            is_fall = True


        #If the figure fall
        else:
            
            figure, state, start, is_fall = place_figure(best_state, best_delta_x, figure, state, num, start, delay)

        #Draw
        draw(WIN, HEIGHT, rows, cols, cell_size, x_offset, y_offset, all_figure, list_colour, colour, figure, next_num)

        #Verify is the end isn't end
        end = end_game(all_figure)

        # Rafraîchit l'affichage
        pygame.display.flip()

    #If the game is over
    else:

        WIN.fill(BLACK)

        FONT = pygame.font.SysFont("trebuchetms", 20)
        GAME_OVER = pygame.font.SysFont("trebuchetms", 50)
        game_over = GAME_OVER.render("Game over", 1, WHITE)

        text_score = FONT.render("Score: " + str(score), 1, WHITE)
        text_level = FONT.render("Level: " + str(level), 1, WHITE)

        WIN.blit(game_over, (WIDTH//2 - game_over.get_width() // 2, HEIGHT//2))
        WIN.blit(text_score, (WIDTH//2 - text_score.get_width()// 2, HEIGHT//2 + game_over.get_height()))
        WIN.blit(text_level, (WIDTH//2 - text_level.get_width()// 2, HEIGHT//2 + text_score.get_height() + game_over.get_height()))

        # Rafraîchit l'affichage
        pygame.display.flip()

# Ferme Pygame
pygame.quit()

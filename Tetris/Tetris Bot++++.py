
import pygame
import random
import numpy as np
import time
import os

pygame.init()

# Crée une fenêtre de 800x600 pixels
WIDTH, HEIGHT = 1910, 1000
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

            if pos[0] == 0:
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
        line_pos = np.dstack((y, x))[0]     #The tab with the coordinate of the line

        #For every cell in the grid
        #You start to the line above that delete, and you go up to line 0
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
                    if bloc[0] < y_line:
                        new_bloc = [bloc[0]+1, bloc[1]]
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
            grid[bloc[0], bloc[1]] = 1
        
    return grid


#Turn the figure
def turn_figure(figure, num, state):
    
    max_y = np.max(figure[:, 0], 0)
    x = int(np.mean(figure[:, 1], 0))


    #Figure1    Tétrimino I
    if num == 1:
        match state:
            case 0:
                # X
                # X
                # X
                # X
                figure = np.array([[max_y-2, x], [max_y-1, x], [max_y, x], [max_y+1, x]])
                new_state = 1
        
            case 1:
                # X X X X
                figure = np.array([[max_y, x-1], [max_y, x], [max_y, x+1], [max_y, x+2]])
                new_state = 0
            

    #Figure2    Tétrimino O
    elif num == 2:
        # X X
        # X X
        figure = figure
        new_state = 0

    #Figure3    Tétrimino T
    elif num == 3:
        match state:
            case 0:
                # X 
                # X X
                # X
                figure = np.array([[max_y-1, x], [max_y, x], [max_y+1, x], [max_y, x+1]])
                new_state = 1


            case 1:
                #   X
                # X X X
                figure = np.array([[max_y-1, x+1], [max_y -1, x], [max_y-1, x-1], [max_y-2, x]])
                new_state = 2
   

            case 2:
                #   X
                # X X
                #   X
                figure = np.array([[max_y-1, x], [max_y, x], [max_y + 1, x], [max_y, x-1]])
                new_state = 3

        
            case 3:
                # X X X
                #   X
                figure = np.array([[max_y, x+2], [max_y, x+1], [max_y, x], [max_y+1, x+1]])
                new_state = 0
 

    #Figure4    Tétrimino L
    elif num == 4:
        match state:
            case 0:
                # X 
                # x
                # x X
                figure = np.array([[max_y-1, x], [max_y, x], [max_y+1, x], [max_y+1, x+1]])
                new_state = 1

        
            case 1:
                #     X
                # X X X
                figure = np.array([[max_y-1, x], [max_y-1, x+1], [max_y-1, x+2], [max_y-2, x+2]])
                new_state = 2

            case 2:
                # X X
                #   X
                #   X
                figure = np.array([[max_y-1, x], [max_y, x], [max_y+1, x], [max_y-1, x-1]])
                new_state = 3

            case 3:
                # X X X
                # X
                figure = np.array([[max_y, x], [max_y, x+1], [max_y, x+2], [max_y+1, x]])
                new_state = 0
    
    #Figure5    Tétrimino J
    elif num == 5:
        match state:
            case 0:
                # X X
                # X
                # X
                figure = np.array([[max_y-1, x], [max_y, x], [max_y+1, x], [max_y-1, x+1]])
                new_state = 1

        
            case 1: 
                # X
                # X X X           
                figure = np.array([[max_y-1, x-1], [max_y-1, x], [max_y-1, x+1], [max_y-2, x-1]])
                new_state = 2

            case 2:
                #   X
                #   X 
                # X X
                figure = np.array([[max_y-1, x+1], [max_y, x+1], [max_y+1, x+1], [max_y+1, x]])
                new_state = 3

            case 3:
                # X X X
                #     X
                figure = np.array([[max_y, x], [max_y, x+1], [max_y, x+2], [max_y+1, x+2]])
                new_state = 0

    #Figure6    Tétrimino Z
    elif num == 6:
        match state:
            case 0:
                #   X
                # X X
                # X
                figure = np.array([[max_y+1, x-1], [max_y, x-1], [max_y, x], [max_y-1, x]])
                new_state = 1
        
            case 1:
                #  X X
                #    X X
                figure = np.array([[max_y, x], [max_y, x+1], [max_y+1, x+1], [max_y+1, x+2]])
                new_state = 0

    #Figure7   Tétrimino S
    elif num == 7:
        match state:
            case 0:
                # X
                # X X
                #   X
                figure = np.array([[max_y-1, x], [max_y, x], [max_y, x+1], [max_y+1, x+1]])
                new_state = 1
        
            case 1:
                #   X X
                # X X
                figure = np.array([[max_y+1, x-1], [max_y+1, x], [max_y, x], [max_y, x+1]])
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
    text_score = FONT.render(f"Score: {score:,}".replace(",", " "), 1, WHITE)
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
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -2], cell_size, marge)
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
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 1], cell_size, marge)


    elif next_num == 4:
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [0, 1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 1], cell_size, marge)

    elif next_num == 5:
        pygame.draw.rect(WIN, BLUE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [0, 1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, -1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, -1], cell_size, marge)

    elif next_num == 6:
        pygame.draw.rect(WIN, RED, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

        #Shader
        draw_shaders(x_offset//2, HEIGHT//2, [0, 0], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [0, 1], cell_size, marge)
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 0], cell_size, marge)
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
        draw_shaders(x_offset//2, HEIGHT//2, [-1, 1], cell_size, marge)


    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2 - 2*cell_size, 4*cell_size, 4*cell_size), 4)


#Draw the figure land
def draw_land_figure(WIN, x_offset, y_offset, gap, all_figure, list_colour, marge):

    for index, figure in enumerate (all_figure):
        for pos in figure:
            pygame.draw.rect(WIN, list_colour[index], (x_offset + pos[1]*gap, y_offset + pos[0]*gap, gap, gap))
            draw_shaders(x_offset, y_offset, pos, gap, marge)


#Draw the figure who fall
def draw_figure(WIN, x_offset, y_offset, gap, colour, figure, marge):

    for pos in figure:
        pygame.draw.rect(WIN, colour, (x_offset + pos[1]*gap, y_offset + pos[0]*gap, gap, gap))
        draw_shaders(x_offset, y_offset, pos, gap, marge)


#Draw the shaders
def draw_shaders(x_offset, y_offset, pos, gap, marge):
            
    #The shaders
    white_poly = ((x_offset + pos[1]*gap, y_offset + pos[0]*gap), (x_offset + (pos[1] + 1)*gap, y_offset + pos[0]*gap), (x_offset + (pos[1] + (1 - marge))*gap, y_offset + (pos[0] + marge)*gap), (x_offset + (pos[1] + marge)*gap, y_offset + (pos[0] + marge)*gap))
    pygame.draw.polygon(WIN, WHITE, white_poly)

    dark_poly = ((x_offset + (pos[1])*gap, y_offset + (pos[0] + 1)*gap), (x_offset + (pos[1] + 1)*gap, y_offset + (pos[0] + 1)*gap), (x_offset + (pos[1] + (1 -marge))*gap, y_offset + (pos[0] + (1 - marge))*gap), (x_offset + (pos[1] + marge)*gap, y_offset + (pos[0] + (1 - marge))*gap))
    pygame.draw.polygon(WIN, SHADOW, dark_poly)

    left_poly = ((x_offset + (pos[1])*gap, y_offset + (pos[0])*gap), (x_offset + (pos[1])*gap, y_offset + (pos[0] +1 )*gap), (x_offset + (pos[1] + marge)*gap, y_offset + (pos[0] + (1 - marge))*gap), (x_offset + (pos[1] + marge)*gap, y_offset + (pos[0] + marge)*gap))
    pygame.draw.polygon(WIN, GREY, left_poly)

    right_poly = ((x_offset + (pos[1]+1)*gap, y_offset + (pos[0])*gap), (x_offset + (pos[1] + 1)*gap, y_offset + (pos[0] + 1)*gap), (x_offset + (pos[1] + (1 - marge))*gap, y_offset + (pos[0] + (1 - marge))*gap), (x_offset + (pos[1] + (1 - marge))*gap, y_offset + (pos[0] + marge)*gap))
    pygame.draw.polygon(WIN, GREY, right_poly)


#Return the new figure 
def add_figure(list_num):

    if len(list_num) == 0:
        list_num = [1, 2, 3, 4, 5, 6, 7]

    rand = random.randint(0, len(list_num)-1)
    num = list_num[rand]      
    list_num.remove(num) 

    #Figure1    Tétrimino I
    match num:
        case 1:
            colour = CYAN
            figure = np.array([[-1, 3], [-1, 4], [-1, 5], [-1, 6]])
            nb_state = 2

    #Figure2    Tétrimino O
        case 2:
            colour = YELLOW
            figure = np.array([[-2, 4], [-1, 4], [-1, 5], [-2, 5]])
            nb_state = 1

    #Figure3    Tétrimino T
        case 3:
            colour = PURPLE
            figure = np.array([[-2, 3], [-2, 4], [-2, 5], [-1, 4]])
            nb_state = 4
        
    #Figure4    Tétrimino L
        case 4:
            colour = ORANGE
            figure = np.array([[-2, 3], [-2, 4], [-2, 5], [-1, 3]])
            nb_state = 4
        
    #Figure5    Tétrimino J
        case 5:
            colour = BLUE
            figure = np.array([[-2, 3], [-2, 4], [-2, 5], [-1, 5]])
            nb_state = 4
        
    #Figure6    Tétrimino Z
        case 6:
            colour = RED
            figure = np.array([[-2, 3], [-2, 4], [-1, 4], [-1, 5]])
            nb_state = 2
    
    #Figure7   Tétrimino S
        case 7:
            colour = GREEN
            figure = np.array([[-1, 3], [-1, 4], [-2, 4], [-2, 5]])
            nb_state = 2
    
    return figure, colour, num, list_num, nb_state


#Update the score
def update_score(score, level, nb_line, tt_nb_line):

    match nb_line:
        case 1:
            score += 40*level

        case 2:
            score += 100*level
            
        case 3:
            score += 300*level

        case  4:
            score += 1200*level

    #Update the level, the level incrise all the 10 lines delete
    tt_nb_line += nb_line
    level = (tt_nb_line // 10) +1

    return score, level, tt_nb_line


def make_score(list_coo, grid, rows, cols):
    
    malus = 0

    #verify if there is a line in game
    for i in range(rows):
            
        #If there is a line
        if all(grid[i,:] == 1):
            grid[i,:] = 0

            #For every cell in the grid
            #You start to the line above that delete, and you go up to line 0
            for row in range(i-1, -1, -1):
                for col in range(cols):
                        
                    #If there is a bloc
                    if grid[row, col] == 1: 
                        
                        #Go down one step
                        grid[row +1 , col] = 1
                        grid[row, col] = 0
    
          
    list_air_bloc = set()
    list_delta = [-1, 1]      

    #For every bloc in the grid
    for bloc in list_coo:
                
        for delta in list_delta:
                    
            if (0 <= bloc[0] + delta < rows):
                if grid[bloc[0] + delta, bloc[1]] == 0:
                            
                    list_air_bloc.add((bloc[0] + delta, bloc[1]))

        for delta in list_delta:
                        
            if (0 <= bloc[1] + delta < cols):
                if grid[bloc[0], bloc[1] + delta] == 0:
                            
                    list_air_bloc.add((bloc[0], bloc[1] + delta))
  
    
    #For every air bloc near a figure in the grid
    malus = 0

    for air_bloc in list_air_bloc:
        nb_neighbors = 0
        for delta in list_delta:
            if (0 <= air_bloc[0] + delta < rows):
                if grid[air_bloc[0] + delta, air_bloc[1]] == 0:
                    nb_neighbors +=1

        for delta in list_delta:
                if (0 <= air_bloc[1] + delta < cols):
                    if grid[air_bloc[0], air_bloc[1] + delta] == 0:
                        nb_neighbors +=1

        if nb_neighbors == 0 :
            malus += 2
        
        elif nb_neighbors == 1:
            malus += 2

    return grid, malus


def find_minimun_Y(list_coo, intervalle_x, rows):

    # Filtrer les coordonnées qui sont dans l'intervalle X
    coords_filtr = [coord for coord in list_coo if intervalle_x[0] <= coord[1] <= intervalle_x[1]]

    # Initialiser un dictionnaire vide
    dict = {}

    # Parcourir chaque paire (y, x) dans coords_filtr, pour regrouper les valeurs Y avec la même coordonnées X
    for y, x in coords_filtr:
        # Si la clé x n'existe pas dans le dictionnaire, crée une nouvelle entrée avec une liste contenant y
        if x not in dict:
            dict[x] = [y]
        else:
            # Si la clé x existe déjà, ajoute y à la liste associée
            dict[x].append(y)

    #Extrait le minimun Y sur l'axes X, s'il n'y a pas de X mets l'indices maximal de l'axe Y
    min_Y = np.array([], np.int8)
    for X in range(intervalle_x[0], intervalle_x[1] + 1):
        if X in dict:
            val = min(dict[X])
        else:
            val = rows
            
        min_Y = np.append(min_Y, [val -1, X])

    min_Y = min_Y.reshape((-1, 2))
    return min_Y


def soustraction(coords1 , coords2):
    # On initialise une liste pour stocker les résultats de la soustraction
    result = []

    # On parcourt chaque paire dans coords1
    for coord1 in coords1:
        y1, x1 = coord1

        # On cherche si un élément dans coords2 a la même valeur de X
        matching_coords = coords2[coords2[:, 1] == x1]
    
        # Si on trouve des correspondances sur le X, on effectue la soustraction
        for coord2 in matching_coords:
            y2, x2 = coord2

            # La soustraction est effectuée sur les Y
            result.append(y1 - y2)  # On conserve le même X

    # Convertir le résultat en tableau numpy
    result = np.array(result)

    return result



def get_figure_coordinates(grid):
    wall_coords = set()

    #As the scann go to top to bottom, first the smallest number of Y are scann then the bigger
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                wall_coords.add((y, x))
    return wall_coords


def find_position(grid, rows, cols, next_figure, figure, next_num, num, nb_state, next_nb_state):
    

    coef_matrix = np.arange(1, rows+1)[::-1]
    coef_matrix = coef_matrix.reshape(rows, 1)
    intial_next_figure = next_figure.copy()
    intial_figure = figure.copy()
    best_score = float("inf")
    
    """"""""""""""
    #First figure
    """"""""""""""
    #Get the coordooné of all figure
    list_coo1 = get_figure_coordinates(grid)
    
    #For every state of the figure
    for state1 in range(nb_state):

        max_x1 = np.max(figure[:, 1], 0) 
        min_x1 = np.min(figure[:, 1], 0)
        
        width_figure1 = max_x1 - min_x1

        #The figure begin to the at the left corner
        figure[:, 1] -= min_x1

        #For every position on the x axis
        for delta_x1 in range(cols - width_figure1):
            
            list_min_Y1 = find_minimun_Y(list_coo1, (delta_x1, delta_x1 + width_figure1 + 1), rows)
            new_pos1 = soustraction(list_min_Y1, figure)

            min_Y1 = np.min(new_pos1)  # Trouver la valeur minimale de la première colonne de b
            nb_contact = np.count_nonzero(new_pos1 == min_Y1)
            result_col1 = figure[:, 0] + min_Y1  # Additionner cette valeur minimale à la première colonne de a

            # Créer le résultat final en empilant verticalement la colonne résultante avec la deuxième colonne de a
            figure1 = np.vstack((result_col1, figure[:, 1])).T

            # For every iteration the figure move one step to the right
            figure[:, 1] += 1

            """"""""""""""
            #Seconde figure
            """"""""""""""
            next_figure = intial_next_figure
            for state2 in range(next_nb_state):

                max_x2 = np.max(next_figure[:, 1], 0) 
                min_x2 = np.min(next_figure[:, 1], 0)
                width_figure2 = max_x2 - min_x2

                #The figure begin to the at the left corner
                next_figure[:, 1] -= min_x2

                #For every position on the x axis
                for delta_x2 in range(cols - width_figure2):


                    #Add the figure1 to the grid and the list of coordonnate
                    grid2 = grid.copy() 
                    list_coo2 = list_coo1.copy() 
                    for bloc1 in figure1:
                        list_coo2.add((bloc1[0] ,  bloc1[1]))
                        grid2[bloc1[0], bloc1[1]] = 1
                    
                    list_min_Y2 = find_minimun_Y(list_coo2, (delta_x2, delta_x2 + width_figure2 + 1), rows)

                    new_pos2 = soustraction(list_min_Y2, next_figure)
                    min_Y2 = np.min(new_pos2)  # Trouver la valeur minimale de la première colonne de b
                    nb_contact += np.count_nonzero(new_pos2 == min_Y2)

                    result_col2 = next_figure[:, 0] + min_Y2 # Additionner cette valeur minimale à la première colonne de a

                    # Créer le résultat final en empilant verticalement la colonne résultante avec la deuxième colonne de a
                    figure2 = np.vstack((result_col2, next_figure[:, 1])).T

                    # For every iteration the figure move one step to the right
                    next_figure[:, 1] += 1

                    #Add the figure2 to the grid and the list of coordonnate
                    for bloc2 in figure2:
                        list_coo2.add((bloc2[0] ,  bloc2[1]))
                        grid2[bloc2[0], bloc2[1]] = 1

                    #Make the score
                    grid2, malus = make_score(list_coo2, grid2, rows, cols)
                    
                    score_matrix = coef_matrix.T.dot(grid2)
                    score = np.sum(score_matrix)
                    score += malus

                    #If the score is lower, save the parametre
                    if score < best_score:
                        best_score = score
                        best_state = state1
                        best_delta_x = delta_x1
                        best_delta_y = min_Y1
                        best_nb_contact = nb_contact
                        

                    #If the score is the same but, the figure has more neighbors, save the parametre
                    elif score == best_score and best_nb_contact < nb_contact:
                        best_score = score
                        best_state = state1
                        best_delta_x = delta_x1
                        best_delta_y = min_Y1
                        best_nb_contact = nb_contact

                next_figure, _ = turn_figure(next_figure, next_num, state2)
        figure, _ = turn_figure(figure, num, state1)
    
    return best_state, best_delta_x, best_delta_y, intial_figure


def place_figure(best_state, best_delta_x, best_delta_y, figure, state, num):

    #Set the state
    while state != best_state:
        figure, state = turn_figure(figure, num, state)           
    
    min_x = np.min(figure[:, 1], 0)

    #Set the x axis
    delta_x = 0
    while min_x + delta_x != best_delta_x:

        if min_x > best_delta_x:
            delta_x -= 1

        else:
            delta_x += 1
        
    new_figure = np.array([], np.int8)
    for bloc in figure:
            
        new_bloc = np.array([], np.int8)
        new_bloc = np.append(new_bloc, bloc[0] + best_delta_y)
        new_bloc = np.append(new_bloc, bloc[1] + delta_x)
        new_figure = np.append(new_figure, new_bloc)

    #Reshape figure because it containt 4 bloc withs 2 coordonates        
    new_figure = new_figure.reshape(4, 2)
    
    return new_figure

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
rows = 20               #How many rows
cols = 10               #How many cols
delay = 0.0  
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

#Play the music
pygame.mixer.music.play(-1)

grid = np.zeros((rows, cols), np.int8)

# Boucle principale
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

    WIN.fill(BLACK)

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
        else:
            #Create the next figure and find the best place the old one
            next_figure, next_colour, next_num, list_num, next_nb_state = add_figure(list_num)
            best_state, best_delta_x, best_delta_y, figure = find_position(grid, rows, cols, next_figure, figure, next_num, num, nb_state, next_nb_state)

            #Place the figure
            figure = place_figure(best_state, best_delta_x, best_delta_y, figure, state, num)

            #Update the grid, the figure landing and the colour
            all_figure.append(figure)
            list_colour.append(colour)
            grid = update_grid(all_figure)

            #Verify if there is a line, and update the game
            all_figure, list_colour, nb_line, grid = verify_line(rows, all_figure, list_colour, grid)
            score, level, tt_nb_line = update_score(score, level, nb_line, tt_nb_line)

            #Pass to the next figure
            figure = next_figure
            colour = next_colour
            num = next_num
            state = 0
            nb_state = next_nb_state


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

        text_score = FONT.render(f"Score: {score:,}".replace(",", " "), 1, WHITE)
        text_level = FONT.render("Level: " + str(level), 1, WHITE)

        WIN.blit(game_over, (WIDTH//2 - game_over.get_width() // 2, HEIGHT//2))
        WIN.blit(text_score, (WIDTH//2 - text_score.get_width()// 2, HEIGHT//2 + game_over.get_height()))
        WIN.blit(text_level, (WIDTH//2 - text_level.get_width()// 2, HEIGHT//2 + text_score.get_height() + game_over.get_height()))

        # Rafraîchit l'affichage
        pygame.display.flip()

# Ferme Pygame
pygame.quit()

import pygame
import random
import numpy as np
import time

pygame.init()

# Crée une fenêtre de 800x600 pixels
WIDTH, HEIGHT = 1500, 800
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

music = pygame.mixer.music.load("Desktop\Document\Programmation\Python\Jeu\Others\Tetris\Tetris_song.mp3")

def end_game(all_figure):
    end = False
    for figure in all_figure:
        for pos in figure:

            if pos[1]==0:
                end = True

    return end

def verify_line(rows, cols, all_figure, list_colour):

    grid = np.zeros((rows, cols), np.int8)

    #Create aa numerical grid tosearch the line
    for figure in all_figure:
        for bloc in figure:
            grid[bloc[1], bloc[0]] = 1
    
    #verify if there is a line in game
    list_y_line = []
    for i in range(rows):

        if all(grid[i,:] == 1):
            list_y_line.append(i)

    #If there is a line in game
    if len(list_y_line) > 0:

        #For every line in game   
        for index_line, y_line in enumerate (list_y_line):


            grid[y_line,:] == 0
            x = np.arange(10)
            y = np.ones((10), np.int8) * y_line
            line_pos = np.dstack((x, y))[0]     #The tab with the coordinate of the line

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
    
    #If there is no line, do nothing
    else : 
        new_all_figure = all_figure
        new_list_colour = list_colour

    nb_line = len(list_y_line)
    return new_all_figure, new_list_colour, nb_line


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
            test_figure = np.array([[x, max_y-2], [x, max_y-1], [x, max_y], [x, max_y+1]])
            state = 1
        
        else:
            # X X X X
            test_figure = np.array([[x-1, max_y], [x, max_y], [x+1, max_y], [x+2, max_y]])
            state = 0

    #Figure2    Tétrimino O
    elif num == 2:
        # X X
        # X X
        test_figure = figure

    #Figure3    Tétrimino T
    elif num == 3:
        if state == 0:
            # X 
            # X X
            # X
            test_figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x+1, max_y]])
            state = 1

        elif state == 1:
            #   X
            # X X X
            test_figure = np.array([[x+1, max_y -1], [x, max_y -1], [x-1, max_y -1], [x, max_y-2]])
            state = 2

        elif state == 2:
            #   X
            # X X
            #   X
            test_figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x-1, max_y]])
            state = 3
        
        else:
            # X X X
            #   X
            test_figure = np.array([[x+2, max_y], [x+1, max_y], [x, max_y], [x+1, max_y+1]])
            state = 0

    #Figure4    Tétrimino L
    elif num == 4:
        if state == 0:
            # X 
            # x
            # x X
            test_figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x+1, max_y+1]])
            state = 1
        
        elif state == 1:
            #     X
            # X X X
            test_figure = np.array([[x, max_y-1], [x+1, max_y-1], [x+2, max_y-1], [x+2, max_y-2]])
            state = 2

        elif state == 2:
            # X X
            #   X
            #   X
            test_figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x-1, max_y-1]])
            state = 3

        else:
            # X X X
            # X
            test_figure = np.array([[x, max_y], [x+1, max_y], [x+2, max_y], [x, max_y+1]])
            state = 0
    
    #Figure5    Tétrimino J
    elif num == 5:
        if state == 0:
            # X X
            # X
            # X
            test_figure = np.array([[x, max_y-1], [x, max_y], [x, max_y+1], [x+1, max_y-1]])
            state = 1
        
        elif state == 1: 
            # X
            # X X X           
            test_figure = np.array([[x-1, max_y-1], [x, max_y-1], [x+1, max_y-1], [x-1, max_y-2]])
            state = 2

        elif state == 2:
            #   X
            #   X 
            # X X
            test_figure = np.array([[x+1, max_y-1], [x+1, max_y], [x+1, max_y+1], [x, max_y+1]])
            state = 3

        else:
            # X X X
            #     X
            test_figure = np.array([[x, max_y], [x+1, max_y], [x+2, max_y], [x+2, max_y+1]])
            state = 0

    #Figure6    Tétrimino Z
    elif num == 6:
        if state == 0:
            #   X
            # X X
            # X
            test_figure = np.array([[x-1, max_y+1], [x-1, max_y], [x, max_y], [x, max_y-1]])
            state = 1
        
        else:
            #  X X
            #    X X
            test_figure = np.array([[x, max_y], [x+1, max_y], [x+1, max_y+1], [x+2, max_y+1]])
            state = 0

    #Figure7   Tétrimino S
    elif num == 7:
        if state == 0:
            # X
            # X X
            #   X
            test_figure = np.array([[x, max_y-1], [x, max_y], [x+1, max_y], [x+1, max_y+1]])
            state = 1
        
        else:
            #   X X
            # X X
            test_figure = np.array([[x-1, max_y+1], [x, max_y+1], [x, max_y], [x+1, max_y]])
            state = 0
        

    max_x = np.max(test_figure[:, 0], 0)
    min_x = np.min(test_figure[:, 0], 0)
    max_y = np.max(test_figure[:, 1], 0)
    contact = False

    #Verefy the colision
    for land_figure in all_figure:
        for bloc_A in land_figure:
            for bloc in test_figure:

                bloc_B= np.array([bloc[0], bloc[1]])
                if np.array_equal(bloc_A, bloc_B):
                    contact = True

    #Add the mouvement    
    if (0 <= min_x and max_x <= 9) and( max_y  < 20) and not contact:
        figure = test_figure

    return figure, state


def move_figure(run, figure, start, delay, all_figure, num, state, start_press, pause, sound_mute):

    delta_x = 0
    delta_y = 0
    fall = 0

    is_fall = True
    contact_x = False
    contact_y = False
    moved = False

    new_figure = np.array([], np.int8)

    if time.time() >= start+ delay:
        start = time.time()
        fall +=1


    #Pygame event
    for event in pygame.event.get():

        #Quit pygame
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.KEYDOWN:
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

    keys = pygame.key.get_pressed()

    #Left
    if keys[pygame.K_LEFT] and time.time() >= (start_press + 0.1):
        start_press = time.time()
        delta_x =  -1
        moved = True

    #Right
    if keys[pygame.K_RIGHT] and time.time() >= (start_press + 0.1):
        start_press = time.time()
        delta_x =  +1
        moved = True
                
    #Down
    if keys[pygame.K_DOWN] and time.time() >= (start_press + 0.1):
        start_press = time.time()
        delta_y =  +1
        moved = True

    #Up
    if keys[pygame.K_UP] and time.time() >= (start_press + 0.2):
        start_press = time.time()
        figure, state = turn_figure(figure, num, state)
        moved = True


    if run:  
        max_x = np.max(figure[:, 0], 0)
        min_x = np.min(figure[:, 0], 0)
        max_y = np.max(figure[:, 1], 0)

        #Verefy the colision
        for land_figure in all_figure:
            for bloc_A in land_figure:
                for bloc in figure:

                    bloc_B= np.array([bloc[0] + delta_x, bloc[1]])
                    if np.array_equal(bloc_A, bloc_B):
                        contact_x = True

                    bloc_B = np.array([bloc[0], bloc[1] + delta_y + fall])
                    if np.array_equal(bloc_A, bloc_B):
                        contact_y = True
        
        if contact_y:
            is_fall = False
                  
        #Add the mouvement
        for bloc in figure:
                
            new_bloc = np.array([], np.int8)
            if (0 <= min_x + delta_x and max_x + delta_x <= 9) and not contact_x:
                new_bloc = np.append(new_bloc, bloc[0] + delta_x)
            else:
                new_bloc = np.append(new_bloc, bloc[0])

            if max_y + delta_y + fall < 20 and not contact_y:
                new_bloc = np.append(new_bloc, bloc[1] + delta_y + fall)
            else:
                new_bloc = np.append(new_bloc, bloc[1])

            new_figure = np.append(new_figure, new_bloc)

        if max_y == 19:
            is_fall = False

        new_figure = new_figure.reshape(4, 2)

    return run, new_figure, start, is_fall, state, start_press, moved, pause, sound_mute
      

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


def draw(WIN, rows, cols, cell_size, x_offset, y_offset, all_figure, list_colour):
    draw_land_figure(WIN, x_offset, y_offset, cell_size, all_figure, list_colour)
    draw_grid(WIN, rows, cols, cell_size, x_offset, y_offset)


def draw_grid(win, rows, cols, cell_size, x_offset, y_offset):

    # Dessiner la grille
    for row in range(rows):
        for col in range(cols):
            x = x_offset + col * cell_size
            y = y_offset + row * cell_size
            pygame.draw.rect(win, WHITE, (x, y, cell_size, cell_size), 1)
            
def draw_next_figure(x_offset, HEIGHT, cell_size, next_num):

    if next_num == 1:
        pygame.draw.rect(WIN, CYAN, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, CYAN, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, CYAN, (x_offset//2 - 2*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, CYAN, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
    
    elif next_num == 2:
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, YELLOW, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))

    elif next_num == 3:
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, PURPLE, (x_offset//2 - 0*cell_size, HEIGHT//2 + 1*cell_size, cell_size, cell_size))
    
    elif next_num == 4:
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, ORANGE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
    
    elif next_num == 5:
        pygame.draw.rect(WIN, BLUE, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, BLUE, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
    
    elif next_num == 6:
        pygame.draw.rect(WIN, RED, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 + 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, RED, (x_offset//2 - 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
    else:
        pygame.draw.rect(WIN, GREEN, (x_offset//2 + 0*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, GREEN, (x_offset//2 + 0*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, GREEN, (x_offset//2 - 1*cell_size, HEIGHT//2 - 0*cell_size, cell_size, cell_size))
        pygame.draw.rect(WIN, GREEN, (x_offset//2 + 1*cell_size, HEIGHT//2 - 1*cell_size, cell_size, cell_size))
        
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

    pygame.draw.rect(WIN, WHITE, (x_offset//2 - 2*cell_size, HEIGHT//2 - 2*cell_size, 4*cell_size, 4*cell_size), 4)


def draw_land_figure(WIN, x_offset, y_offset, gap, all_figure, list_colour):

    for index, figure in enumerate (all_figure):
        for pos in figure:
            pygame.draw.rect(WIN, list_colour[index], (x_offset + pos[0]*gap, y_offset+pos[1]*gap, gap, gap))


def draw_figure(WIN, x_offset, y_offset, gap, colour, figure):

    for pos in figure:
        pygame.draw.rect(WIN, colour, (x_offset + pos[0]*gap, y_offset+pos[1]*gap, gap, gap))
        


def add_figure(list_num):

    if len(list_num) == 0:
        list_num = [1, 2, 3, 4, 5, 6, 7]

    rand = random.randint(0, len(list_num)-1)
    num = list_num[rand]      
    list_num.remove(num) 
    state = 0

    #Figure1    Tétrimino I
    if num == 1:
        colour = CYAN
        figure = np.array([[3, 0], [4, 0], [5, 0], [6, 0]])

    #Figure2    Tétrimino O
    elif num == 2:
        colour = YELLOW
        figure = np.array([[4, -1], [4, 0], [5, 0], [5, -1]])

    #Figure3    Tétrimino T
    elif num == 3:
        colour = PURPLE
        figure = np.array([[3, -1], [4, -1], [5, -1], [4, 0]])
        
    #Figure4    Tétrimino L
    elif num == 4:
        colour = ORANGE
        figure = np.array([[3, -1], [4, -1], [5, -1], [3, 0]])
        
    #Figure5    Tétrimino J
    elif num == 5:
        colour = BLUE
        figure = np.array([[3, -1], [4, -1], [5, -1], [5, 0]])
        
    #Figure6    Tétrimino Z
    elif num == 6:
        colour = RED
        figure = np.array([[3, -1], [4, -1], [4, 0], [5, 0]])
    
    #Figure7   Tétrimino S
    elif num == 7:
        colour = GREEN
        figure = np.array([[3, 0], [4, 0], [4, -1], [5, -1]])
    
    return figure, colour, num, state, list_num

#Bool
run = True
is_fall = True
end = False
new_figure = True
pause = False
sound_mute = False

#List
list_num = [1, 2, 3, 4, 5, 6, 7]
list_colour = []
all_figure = []

#Int
marge = 0
rows = 20
cols = 10
time_delay = 0.5
delay = time_delay
start_press = 0
score = 0
level = 1
tt_nb_line = 0

#Time
start_new_figure = time.time() - delay

x_offset, y_offset, cell_size = dim_grid(WIDTH, HEIGHT, rows, cols, marge)

figure, colour, num, state, list_num = add_figure(list_num)
next_figure, next_colour, next_num, next_state, list_num = add_figure(list_num)

# Boucle principale
clock = pygame.time.Clock()
start = time.time()

pygame.mixer.music.play(-1)

while run:
            
    #Pygame event
    for event in pygame.event.get():
            
        #Quit pygame
        if event.type == pygame.QUIT:
            run = False


        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False

            if event.key ==  pygame.K_SPACE:
                if pause:
                    pause = not pause
                
                else:
                    pause = not pause
            
            if event.key ==  pygame.K_m:
                if sound_mute:
                    pygame.mixer.music.unpause()
                    sound_mute = not sound_mute
                else:
                    pygame.mixer.music.pause()
                    sound_mute = not sound_mute

    clock.tick(60)
    WIN.fill(BLACK)

    if not end:
        
        if pause:

            draw_figure(WIN, x_offset, y_offset, cell_size, colour, figure)
            draw_next_figure(x_offset, HEIGHT, cell_size, next_num)
            draw(WIN, rows, cols, cell_size, x_offset, y_offset, all_figure, list_colour)

            FONT = pygame.font.SysFont("trebuchetms", 20)
            text_score = FONT.render("Score: " + str(score), 1, WHITE)
            text_level = FONT.render("Level: " + str(level), 1, WHITE)
            WIN.blit(text_score, (WIDTH//8, 5*HEIGHT//6))
            WIN.blit(text_level, (WIDTH//8, 5*HEIGHT//6 - text_score.get_height()))
            
            PAUSE = pygame.font.SysFont("trebuchetms", 50)
            text_pause = PAUSE.render("PAUSE", 1, WHITE)
            WIN.blit(text_pause, (WIDTH//2 - text_pause.get_width()//2, HEIGHT//2 - text_pause.get_height()//2))

        elif not is_fall and time.time() >= (start_new_figure + delay):
            
            all_figure.append(figure)
            list_colour.append(colour)
        
            all_figure, list_colour, nb_line = verify_line(rows, cols, all_figure, list_colour)

            figure = next_figure
            colour = next_colour
            num = next_num
            state = next_state

            if nb_line == 1:
                score += 40*level

            elif nb_line == 2:
                score += 100*level
            
            elif nb_line == 3:
                score += 300*level

            elif nb_line == 4:
                score += 1200*level

            tt_nb_line += nb_line
            level = (tt_nb_line // 10) +1
            delay = time_delay - (level - 1) * 0.1

            next_figure, next_colour, next_num, next_state, list_num = add_figure(list_num)
            start = time.time()
            new_figure = True
            is_fall = True

        else:
            run, figure, start, is_fall, state, start_press, moved, pause, sound_mute = move_figure (run, figure, start, delay, all_figure, num, state, start_press, pause, sound_mute)
            draw_figure(WIN, x_offset, y_offset, cell_size, colour, figure)

            if not is_fall and new_figure:
                start_new_figure = time.time()
                new_figure = False
        
        draw_next_figure(x_offset, HEIGHT, cell_size, next_num)
        draw(WIN, rows, cols, cell_size, x_offset, y_offset, all_figure, list_colour)
        end = end_game(all_figure)

        FONT = pygame.font.SysFont("trebuchetms", 20)
        text_score = FONT.render("Score: " + str(score), 1, WHITE)
        text_level = FONT.render("Level: " + str(level), 1, WHITE)
        WIN.blit(text_score, (x_offset//2 - text_score.get_width()//2, 3*HEIGHT//4))
        WIN.blit(text_level, (x_offset//2 - text_score.get_width()//2, 3*HEIGHT//4 - text_score.get_height()))

        # Rafraîchit l'affichage
        pygame.display.flip()

    else:
        WIN.fill(BLACK)
        GAME_OVER = pygame.font.SysFont("trebuchetms", 50)
        game_over = GAME_OVER.render("Game over", 1, WHITE)
        WIN.blit(game_over, (WIDTH//2 - game_over.get_width() // 2, HEIGHT//2))
        WIN.blit(text_score, (WIDTH//2 - text_score.get_width()// 2, HEIGHT//2 + game_over.get_height()))
        WIN.blit(text_level, (WIDTH//2 - text_level.get_width()// 2, HEIGHT//2 + text_score.get_height() + game_over.get_height()))

        # Rafraîchit l'affichage
        pygame.display.flip()

# Ferme Pygame
pygame.quit()

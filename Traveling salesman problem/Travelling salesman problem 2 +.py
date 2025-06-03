import pygame
import numpy as np
import math
import sys

pygame.init()

def solve_tsp(WIN, list_point, order_point, distance, path, start_point, previous_point, nb_combinaison, best_distance, best_order, initial_list, name_list, total_combinaison):

    """
    WIN : the window to draw
    list_point numpy.array: list of point doesn't yet visited
    order_point numpy.array: list of the order of the point
    distance float: distance already calculated
    path numpy.array: path use currently
    start_point numpy.array[numpy.array]: coordonné of the first point
    previous_point numpy.array: coordonné of the last point
    nb_combinaison int: count how many combinaison remaining
    best_distance float: shortess distance caluleted
    best_order numpy.array: best_order
    """
    
    #If we have done all the combinasion, left the function
    if nb_combinaison == 0:
        return nb_combinaison, best_distance, best_order
    
    #Else for every point remnaing
    for index, point in enumerate (list_point):
        
        #Calcul the distance
        d = np.sqrt((point[0] - previous_point[0])**2 + (point[1] - previous_point[1])**2)

        #Update the distance and the path
        new_distance = distance + d
        new_path = np.append(path, order_point[index])

        #Create a mask to keep only the point not visited
        mask = np.arange(len(list_point))
        new_list_point = list_point[mask != index]
        new_order_point = order_point[mask != index]
                
        #If we have visited all the point
        if len(new_list_point) == 0:
            
            #Verify if the last digit is greater than the second, to avoid counting double
            if new_path[-1] >  new_path[1]:
                nb_combinaison -=1
            
                #Calcul the distance between the last point and the fisrt point
                d = np.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                new_distance += d

                #If the distance calculted is below the best distance swich the variable
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_order = new_path

                #Draw_line
                pg_event()
                WIN.blit(background_surface, (0, 0))
                draw_line(WIN, initial_list, new_path, WHITE, 3)
                draw_line(WIN, initial_list, best_order, "magenta", 3)
                draw_best_distance(best_distance, nb_combinaison, total_combinaison)
                pygame.display.flip()
            

        #Call the function recursively
        nb_combinaison, best_distance, best_order = solve_tsp(WIN, new_list_point, new_order_point, new_distance, new_path.astype(int), start_point, list_point[index], nb_combinaison, best_distance, best_order.astype(int), initial_list, name_list, total_combinaison)
        
    return nb_combinaison, best_distance, best_order


def draw_point(WIN, list_point, liste_name):
    for point, name in zip(list_point, liste_name):
        pygame.draw.circle(WIN, WHITE, (point[0], point[1]), 5)

        font = pygame.font.Font(None, 50)
        text_surface = font.render(name, True, WHITE)
        text_position = (point[0]-15, point[1])  # 10px de marge
        WIN.blit(text_surface, text_position)


def draw_line(WIN, list_point, order, colour, linewidth):
        pygame.draw.lines(WIN, colour, True, list_point[order], linewidth)


def draw_best_distance(best_distance, nb_combinaison, total_combinaison):
    # Création de la surface de texte
    font = pygame.font.Font(None, 30)
    text_surface1 = font.render("Best distance {:,.0f}".format(best_distance).replace(",", " "), True, WHITE)
    text_position1 = (10, HEIGHT - text_surface1.get_height() - 10)  # 10px de marge
    WIN.blit(text_surface1, text_position1)

    # Création de la surface de texte
    ratio = (1-(nb_combinaison/total_combinaison))*100
    text_surface2 = font.render("Progress: {:.1f}%".format(ratio), True, WHITE)
    text_position2 = (10, HEIGHT - text_surface2.get_height() - text_surface1.get_height() - 10)  # 10px de marge
    WIN.blit(text_surface2, text_position2)

def pg_event():

    #Pygame event
    for event in pygame.event.get():
            
        #Quit pygame
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()


def get_number_input(WIN):

    # Gestion des événements
    run = True
    user_input = ""
    while run:
        
        WIN.fill(BLACK)
        # Dessiner une boîte pour l'entrée
        input_box = pygame.Rect(WIDTH - 200, 10, 200, 50)
        pygame.draw.rect(WIN, WHITE, input_box, 2)

        # Afficher l'entrée de l'utilisateur
        font = pygame.font.Font(None, 50)
        input_text = font.render(user_input, True, WHITE)
        WIN.blit(input_text, (input_box.x + 10, input_box.y + 10))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER :  # Valider avec la touche Entrée
                    try:
                        user_number = int(user_input)  # Convertir l'entrée en nombre
                        if user_number > 2:
                            run = False
                        else:
                            print("Erreur : Veuillez entrer un nombre supérieur à 2.")
                            user_input = ""

                    except ValueError:
                        print("Erreur : Veuillez entrer un nombre valide.")
                        user_input = ""

                elif event.key == pygame.K_BACKSPACE:  # Effacer le dernier caractère
                    user_input = user_input[:-1]

                else:
                    # Ajouter le caractère à l'entrée utilisateur
                    if event.unicode.isdigit():  # Limiter aux chiffres
                        user_input += event.unicode

            pygame.display.update()

    return user_number

# Crée une fenêtre de 800x600 pixels
WIDTH, HEIGHT = 1500, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
background_surface = pygame.Surface(WIN.get_size())  # Taille de la fenêtre

# Définit la couleur blanche
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
GREEN = [0, 255, 0]

# Remplit l'écran avec la couleur blanche
WIN.fill(BLACK)

# Définit le titre de la fenêtre
pygame.display.set_caption("Basic pygame")

# Boucle principale
find_shortess_path = False
while True:
            

    #Pygame event
    for event in pygame.event.get():
            
        #Quit pygame
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if event.key == pygame.K_RETURN  and find_shortess_path:
                find_shortess_path = False

    if not find_shortess_path:
         
        number_point = get_number_input(WIN)
        print(number_point)

        X = np.random.randint(0, WIDTH, number_point)
        Y = np.random.randint(0, HEIGHT - 20, number_point)

        #Input
        list_point = np.stack((X, Y), axis=-1)
        order_point = np.arange(number_point)
        path = np.array([])
        start_point =  list_point[0]
        previous_point = list_point[0]
        nb_combinaison = math.factorial(number_point-1)/2 
        best_distance = np.inf
        best_order = np.array([], dtype="i")
        initial_list = list_point
        name_list = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZ"])[0][:number_point]
        
        #Draw
        WIN.fill(BLACK)
        background_surface.fill(BLACK)  # Remplir de noir
        draw_point(background_surface, list_point, name_list)
        WIN.blit(background_surface, (0, 0))
        
        print("nb combinaison", "{:,.0f}".format(nb_combinaison).replace(",", " "))
        _, best_distance, best_order = solve_tsp(WIN, list_point, order_point, 0, path, start_point, previous_point, nb_combinaison, best_distance, best_order, initial_list, name_list, int(nb_combinaison))

        
        print("distance ",best_distance)
        print("path:",''.join([name_list[i] for i in best_order]))
        print("")

        find_shortess_path = True

        #Draw
        WIN.fill(BLACK)
        WIN.blit(background_surface, (0, 0))
        draw_line(WIN, initial_list, best_order, GREEN, 5)
        draw_point(WIN, list_point, name_list)
        draw_best_distance(best_distance, 0, 1)
        

    # Rafraîchit l'affichage
    pygame.display.flip()



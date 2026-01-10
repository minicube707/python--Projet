import numpy as np
import math
import time

def solve_tsp(list_point, order_point, distance, path, start_point, previous_point, nb_combinaison, best_distance, best_order, total_combinaison):

    """
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
                ratio = (1-(nb_combinaison/total_combinaison))*100
                print('{:.1f}%'.format(ratio), end="\r")

                #Calcul the distance between the last point and the fisrt point
                d = np.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                new_distance += d

                #If the distance calculted is below the best distance swich the variable
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_order = new_path

        #Call the function recursively
        nb_combinaison, best_distance, best_order = solve_tsp(new_list_point, new_order_point, new_distance, new_path, start_point, list_point[index], nb_combinaison, best_distance, best_order.astype(int), total_combinaison)
        
    return nb_combinaison, best_distance, best_order



WIDTH, HEIGHT = 300, 300
number_point = int(input("How many point do you want ?\n"))
while number_point <= 2:
    print("")
    print("Enter a number greater than 2")
    number_point = int(input("\nHow many point do you want ?\n"))
    
X = np.random.randint(0, WIDTH, number_point)
Y = np.random.randint(0, WIDTH, number_point)

#Input
list_point = np.stack((X, Y), axis=-1)
order_point = np.arange(number_point)
path = np.array([])
start_point =  list_point[0]
previous_point = list_point[0]
nb_combinaison = math.factorial(number_point-1)/2
best_distance = np.inf
best_order = np.array([], dtype="i")

print("nb combinaison", "{:,.0f}".format(nb_combinaison).replace(",", " "))
start = time.time()

_, best_distance, best_order = solve_tsp(list_point, order_point, 0, path, start_point, previous_point, int(nb_combinaison), best_distance, best_order, int(nb_combinaison))
stop = time.time()
print('{:.2f}%'.format(100))
print("Time: ", stop - start)

print("")
print("distance ",best_distance)
print("path \n",list_point[best_order])

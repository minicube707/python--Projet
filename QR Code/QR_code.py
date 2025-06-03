import numpy as np
import pygame
import math

np.set_printoptions(linewidth=150)

#White = 0
#Black = 1

WHITE= (255, 255, 255)
BLACK= (0, 0, 0)

HEIGHT = 800
WIDTH = 800


#The first colomn is exponnet 
#The second  colomn is log
log_table = np.array([
        [  1,   0],
        [  2,   0],
        [  4,   1],
        [  8,  25],
        [ 16,   2],
        [ 32,  50],
        [ 64,  26],
        [128, 198],
        [ 29,   3],
        [ 58, 223],
        [116,  51],
        [232, 238],
        [205,  27],
        [135, 104],
        [ 19, 199],
        [ 38,  75],
        [ 76,   4],
        [152, 100],
        [ 45, 224],
        [ 90,  14],
        [180,  52],
        [117, 141],
        [234, 239],
        [201, 129],
        [143,  28],
        [  3, 193],
        [  6, 105],
        [ 12, 248],
        [ 24, 200],
        [ 48,   8],
        [ 96,  76],
        [192, 113],
        [157,   5],
        [ 39, 138],
        [ 78, 101],
        [156,  47],
        [ 37, 225],
        [ 74,  36],
        [148,  15],
        [ 53,  33],
        [106,  53],
        [212, 147],
        [181, 142],
        [119, 218],
        [238, 240],
        [193,  18],
        [159, 130],
        [ 35,  69],
        [ 70,  29],
        [140, 181],
        [  5, 194],
        [ 10, 125],
        [ 20, 106],
        [ 40,  39],
        [ 80, 249],
        [160, 185],
        [ 93, 201],
        [186, 154],
        [105,   9],
        [210, 120],
        [185,  77],
        [111, 228],
        [222, 114],
        [161, 166],
        [ 95,   6],
        [190, 191],
        [ 97, 139],
        [194,  98],
        [153, 102],
        [ 47, 221],
        [ 94,  48],
        [188, 253],
        [101, 226],
        [202, 152],
        [137,  37],
        [ 15, 179],
        [ 30,  16],
        [ 60, 145],
        [120,  34],
        [240, 136],
        [253,  54],
        [231, 208],
        [211, 148],
        [187, 206],
        [107, 143],
        [214, 150],
        [177, 219],
        [127, 189],
        [254, 241],
        [225, 210],
        [223,  19],
        [163,  92],
        [ 91, 131],
        [182,  56],
        [113,  70],
        [226,  64],
        [217,  30],
        [175,  66],
        [ 67, 182],
        [134, 163],
        [ 17, 195],
        [ 34,  72],
        [ 68, 126],
        [136, 110],
        [ 13, 107],
        [ 26,  58],
        [ 52,  40],
        [104,  84],
        [208, 250],
        [189, 133],
        [103, 186],
        [206,  61],
        [129, 202],
        [ 31,  94],
        [ 62, 155],
        [124, 159],
        [248,  10],
        [237,  21],
        [199, 121],
        [147,  43],
        [ 59,  78],
        [118, 212],
        [236, 229],
        [197, 172],
        [151, 115],
        [ 51, 243],
        [102, 167],
        [204,  87],
        [133,   7],
        [ 23, 112],
        [ 46, 192],
        [ 92, 247],
        [184, 140],
        [109, 128],
        [218,  99],
        [169,  13],
        [ 79, 103],
        [158,  74],
        [ 33, 222],
        [ 66, 237],
        [132,  49],
        [ 21, 197],
        [ 42, 254],
        [ 84,  24],
        [168, 227],
        [ 77, 165],
        [154, 153],
        [ 41, 119],
        [ 82,  38],
        [164, 184],
        [ 85, 180],
        [170, 124],
        [ 73,  17],
        [146,  68],
        [ 57, 146],
        [114, 217],
        [228,  35],
        [213,  32],
        [183, 137],
        [115,  46],
        [230,  55],
        [209,  63],
        [191, 209],
        [ 99,  91],
        [198, 149],
        [145, 188],
        [ 63, 207],
        [126, 205],
        [252, 144],
        [229, 135],
        [215, 151],
        [179, 178],
        [123, 220],
        [246, 252],
        [241, 190],
        [255,  97],
        [227, 242],
        [219,  86],
        [171, 211],
        [ 75, 171],
        [150,  20],
        [ 49,  42],
        [ 98,  93],
        [196, 158],
        [149, 132],
        [ 55,  60],
        [110,  57],
        [220,  83],
        [165,  71],
        [ 87, 109],
        [174,  65],
        [ 65, 162],
        [130,  31],
        [ 25,  45],
        [ 50,  67],
        [100, 216],
        [200, 183],
        [141, 123],
        [  7, 164],
        [ 14, 118],
        [ 28, 196],
        [ 56,  23],
        [112,  73],
        [224, 236],
        [221, 127],
        [167,  12],
        [ 83, 111],
        [166, 246],
        [ 81, 108],
        [162, 161],
        [ 89,  59],
        [178,  82],
        [121,  41],
        [242, 157],
        [249,  85],
        [239, 170],
        [195, 251],
        [155,  96],
        [ 43, 134],
        [ 86, 177],
        [172, 187],
        [ 69, 204],
        [138,  62],
        [  9,  90],
        [ 18, 203],
        [ 36,  89],
        [ 72,  95],
        [144, 176],
        [ 61, 156],
        [122, 169],
        [244, 160],
        [245,  81],
        [247,  11],
        [243, 245],
        [251,  22],
        [235, 235],
        [203, 122],
        [139, 117],
        [ 11,  44],
        [ 22, 215],
        [ 44,  79],
        [ 88, 174],
        [176, 213],
        [125, 233],
        [250, 230],
        [233, 231],
        [207, 173],
        [131, 232],
        [ 27, 116],
        [ 54, 214],
        [108, 244],
        [216, 234],
        [173, 168],
        [ 71,  80],
        [142,  88],
        [  1, 175]])

def make_forbidden_cell(len_qr_code):

    #Top left square
    # Créer un maillage de coordonnées
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.array([], np.int8)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #Top right square
    # Créer un maillage de coordonnées
    x, y = np.meshgrid(np.arange(len_qr_code-8, len_qr_code), np.arange(8))
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #Bottom left square
    # Créer un maillage de coordonnées
    x, y = np.meshgrid(np.arange(8), np.arange(len_qr_code-8, len_qr_code))
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #Format information
    #1
    x, y = np.meshgrid(np.arange(9), np.ones(9)*8)
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #2
    x, y = np.meshgrid(np.ones(8)*8, np.arange(8))
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #3
    x, y = np.meshgrid(np.arange(len_qr_code-8, len_qr_code), np.ones(9)*8)
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #4
    x, y = np.meshgrid(np.ones(9)*8, np.arange(len_qr_code-8, len_qr_code))
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #Timng 
    #Vertical timing
    x, y = np.meshgrid(np.ones(len_qr_code-17)*6, np.arange(len_qr_code-17, len_qr_code-8))
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))

    #Horizontal timing
    x, y = np.meshgrid(np.arange(len_qr_code-17, len_qr_code-8), np.ones(len_qr_code-17)*6, )
    # Empiler les deux tableaux pour obtenir les coordonnées sous forme (x, y)
    forbidden_cell = np.append(forbidden_cell, np.stack([x, y], axis=-1))
    forbidden_cell = forbidden_cell.reshape((-1, 2)).astype(np.int8)

    return forbidden_cell


def up(grid, list, index, coo_start, coo_stop, forbidden_cell):
    x = coo_start[0]
    y = coo_start[1]
    
    #If the cell isn't forbbiden, writte
    if not any(np.array_equal(sub, np.array([x, y]))  for sub in forbidden_cell):
        grid[y, x] = list[index]
        index +=1

    i = 0
    while (x, y) != coo_stop:
        if i % 2 ==0:
            x -=1
        
        else:
            x +=1
            y -=1

        i +=1
        #If the cell isn't forbbiden, writte
        if not any(np.array_equal(sub, np.array([x, y]))  for sub in forbidden_cell):
            grid[y, x] = list[index]
            index +=1

    return grid, index


def down(grid, list, index, coo_start, coo_stop, forbidden_cell):
    x = coo_start[0]
    y = coo_start[1]

    if not any(np.array_equal(sub, np.array([x, y]))  for sub in forbidden_cell):
        grid[y, x] = list[index]
        index +=1

    i = 0
    while (x, y) != coo_stop:
        if i % 2 ==0:
            x -=1
        
        else:
            x +=1
            y +=1

        i +=1
        
        #If the cell isn't forbbiden, writte
        if not any(np.array_equal(sub, np.array([x, y]))  for sub in forbidden_cell):
            grid[y, x] = list[index]
            index +=1

    return grid, index


def make_colonn(grid, n , index, point_a, point_b, forbiden_cell):

    #X
    if point_a[0] < point_b[0]:
        max_x = point_b[0]
        min_x = point_a[0]
    
    else:
        min_x = point_b[0]
        max_x = point_a[0]
    
    #Y
    if point_a[1] < point_b[1]:
        max_y = point_b[1]
        min_y = point_a[1]
    
    else:
        min_y = point_b[1]
        max_y = point_a[1]
    
    
    grid, index = up(grid, n, index,  (max_x, max_y), (max_x-1, min_y), forbiden_cell)
    grid, index = down(grid, n, index,  (max_x-2, min_y), (min_x, max_y), forbiden_cell)
    
    return grid, index-1


def make_QR_Code(len_qr_code, final_message, forbidden_cell):

    grid = np.zeros((len_qr_code, len_qr_code), np.int16)
    index = -1
    
    for i in range(((len_qr_code-1)//4)-2):
        
        coor_column1 = (len_qr_code -(i+1)*4, 0)
        coor_column2 = (len_qr_code -1-(i)*4, len_qr_code-1)
    
        grid, index = make_colonn(grid, final_message, index+1, coor_column1, coor_column2, forbidden_cell)

    grid, index = up(grid, final_message, index+1, (8, len_qr_code-8), (7, 9), forbidden_cell)
    grid, index = down(grid, final_message, index, (5, 9), (4, len_qr_code-8), forbidden_cell)
    grid, index = make_colonn(grid, final_message, index, (3, len_qr_code-8), (0, 9), forbidden_cell)

    grid[6, ::2] += 1
    grid[::2, 6] += 1
    grid[6, 6] = 1

    #Finder patterns
    #Top Left
    grid[:8, :8] = 0
    grid[:7, :7] = 1
    grid[1:6, 1:6] = 0
    grid[2:5, 2:5] = 1

    #Top right
    grid[:8, 13:] = 0
    grid[:7, 14:] = 1
    grid[1:6, 15:20] = 0
    grid[2:5, 16:19] = 1

    #Bottom Left
    grid[13:, :8] = 0
    grid[14: , :7] = 1
    grid[15:20, 1:6] = 0
    grid[16:19, 2:5] = 1
    
    #Dark module
    grid[13, 8] = 1
    
    return grid


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


#Draw the grid
def draw_grid(win, rows, cols, cell_size, x_offset, y_offset, grid):

    # Dessiner la grille
    for row in range(rows):
        for col in range(cols):
            x = x_offset + col * cell_size
            y = y_offset + row * cell_size
            #pygame.draw.rect(win, BLACK, (x, y, cell_size, cell_size), 1)

            if grid[row, col] == 1:
                pygame.draw.rect(win, BLACK, (x, y, cell_size, cell_size))


def mask(num_mask, i , j):
    match num_mask:
        case 0:
            return (i+j)%2 == 0

        case 1:
            return i%2 == 0
    
        case 2:
            return j%3 == 0
    
        case 3:
            return (i+j)%3 == 0
    
        case 4:
            return (math.floor(i / 2) + math.floor(j / 3)) % 2 == 0
                    
        case 5:
            return (i*j)%2 + (i*j)%3 == 0
        
        case 6:
            return ((i * j) % 2 + (i * j) % 3) % 2 == 0
                    
        
        case 7:
            return ((i * j) % 3 + (i + j) % 2) % 2 == 0
        


def evalutaion_1(grid):
    n = len(grid)
    penalty = 0
    new = True
    
    # Vérifier les lignes
    for i in range(n):
        for j in range(n - 4):  # on ne vérifie que jusqu'à n-5 inclus
            if np.all(grid[i, j:j+5] == 1):  # Vérifie si 5 consécutifs dans la ligne
                
                #If it's the first add 3 to the penality
                if new:
                    penalty += 3
                    new = False
                
                #Else add 1
                else:
                    penalty +=1

            else:
                new = True
    
    # Vérifier les colonnes
    new = True
    for j in range(n):
        for i in range(n - 4):
            if np.all(grid[i:i+5, j] == 1):  # Vérifie si 5 consécutifs dans la colonne

                #If it's the first add 3 to the penality
                if new:
                    penalty += 3
                    new = False

                #Else add 1
                else:
                    penalty +=1

            else:
                new = True

    return penalty


def evalutaion_2(grid):

    n = len(grid)
    penalty = 0
  
    # Vérifier les lignes
    for i in range(n-1):
        for j in range(n - 1):  # on ne vérifie que jusqu'à n-5 inclus
            if np.all(grid[i:i+2, j:j+2] == 1) or np.all(grid[i:i+2, j:j+2] == 0):
                penalty += 3

    return penalty


def evalutaion_3(grid):

    n = len(grid)
    penalty = 0

    arr_1 = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0])
    arr_2 = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1])

    for i in range(n):
        for j in range(n - 9):  
            if np.array_equal(arr_1, grid[i, j:j+10]) or np.array_equal(arr_2, grid[i, j:j+10]):
                penalty += 40
    
    for i in range(n):
        for j in range(n - 9): 
            if np.array_equal(arr_1, grid[j:j+10, i]) or np.array_equal(arr_2, grid[j:j+10, i]):
                penalty += 40

    return penalty


def evalutaion_4(grid):

    nb_cel = len(grid)**2 #total of cell
    nb_dark = np.count_nonzero(grid) #all dark cell
    percent = (nb_dark / nb_cel) * 100

    #Take the smalest multipe of 5
    a = percent//5
    b = percent//5 +5

    #Sub by 50
    a = np.abs(50 -a)
    b = np.abs(50 -b)

    #Divide by 5
    a //= 5
    b //= 5

    if a < b:
        return a*10
    else:
        return b*10


def num_encode(message):
    bit_message = []

    #For all the 3 digits
    for i in range(0, len(message), 3):

        #Verify if the fisrt digits begin with a 0
        if int(message[i]) == 0:

            #Verify if the seconde digits begin with a 0
            if int(message[i+1]) == 0:
                bit_message.append(format(int(message[i+2:i+3]), "04b"))
                print(int(message[i+2:i+3]))
                
            else:
                bit_message.append(format(int(message[i+1:i+3]), "07b"))
                print(int(message[i+1:i+3]))
                
        else:
            if len(message[i:i+3]) == 3:
                bit_message.append(format(int(message[i:i+3]), "010b"))
                print(int(message[i:i+3]))
                
            elif len(message[i:i+3]) == 2:
                bit_message.append(format(int(message[i:i+3]), "07b"))
                print(int(message[i:i+3]))
                
            else:
                bit_message.append(format(int(message[i:i+3]), "04b"))
                print(int(message[i:i+3]))

    return bit_message


def alpha_enode(message):
    bit_message = []
    alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    
    print("\nAlpha encode")
    #For 2 digits
    for i in range(0, len(message)-1, 2):
        a = message[i]
        b = message[i+1]

        index_a = alphanumeric.index(a)
        index_b = alphanumeric.index(b)

        num = (index_a *45) + index_b
        bit_message.append(format(num, "011b"))
        print(num)

    #If the message has odd digits
    if len(message)%2 == 1:
        c = message[-1]
        index_c = alphanumeric.index(c)
        bit_message.append(format(index_c, "06b"))
        print(num)

    return bit_message


def bit_encode(message):

    bit_message = []
    for i in range(len(message)):
        bit_message.append(format(ord(message[i]), "08b"))

    return bit_message


def encode(message,  total_data_codewords):

    numeric = "0123456789"
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789$%*+-./:"

    count = len(message)

    #Select the mode
    if all(digit in list(numeric) for digit in list(message)):
        bit_count = format(count, "010b")
        mode = 1
        print("mode: numeric")
    
    elif all(digit in list(alphanumeric) for digit in list(message)):
        bit_count = format(count, "09b")
        mode = 2
        print("mode: alphanumeric")
    
    else:
        bit_count = format(count, "08b")
        mode = 4
        print("mode: bit")

    #Encode    
    if mode == 1:
        bit_message = num_encode(message)

    elif mode == 2:
        bit_message = alpha_enode(message)

    else:
        bit_message = bit_encode(message)

    #Add the segment mode, then the segment count and finish by the segment terùinaison
    bit_message.insert(0, format(mode, "04b"))
    bit_message.insert(1, bit_count)
    bit_message.append(format(0, "04b"))
    
    #Transform the list in a string
    new_bit_message = ""
    for bit in bit_message:
        new_bit_message += bit
    
    #Add the number of zeros to have a message in octect
    if len(new_bit_message)%8 != 0:
       new_bit_message += (8 - len(new_bit_message)%8)*"0"

    #Fill the code_word to have exactly the good number of octet
    diff = total_data_codewords*8 - len(new_bit_message)
    for i in range(diff//8):
        if i%2==0:
            new_bit_message += "11101100"
        
        else:
            new_bit_message += "00010001"

    #Transfom the binary number to decimal number
    final_message = np.array([], np.int8)
    for i in range(0, len(new_bit_message), 8):
        final_message = np.append(final_message, int(new_bit_message[i:i+8], 2))   

    return final_message


def poly_multi(poly_1, poly_2, log_table):

    #Initialise la longeur du polynome finale
    res_poly = np.zeros(len(poly_1) + len(poly_2) -1, np.int16)
    
    for i in range(len(poly_1)):
        coef_1 = poly_1[i]
        for j in range(len(poly_2)):
            coef_2 = poly_2[j]    

            #Add the exponent, apply the modulo        
            num = (coef_1 + coef_2)%255

            #Then Xor the log of the add 
            res_poly[i+j] ^= log_table[num][0]

    #Retrun the exponent for each coef of the polynome
    res_poly = log_table[res_poly][:, 1]
    return res_poly


def poly_div_euc(poly_1, poly_2):

    poly_coef = np.array([])
    poly_4 = poly_1[0:len(poly_2)]

    for i in range(len(poly_1) - len(poly_2)+1):

        coef = poly_4[0]/poly_2[0]
        poly_coef = np.append(poly_coef, coef)
        poly_3 = poly_2 * coef

        subtraction = poly_4 - poly_3
        poly_4 = subtraction[1:]

        if i < len(poly_1)- len(poly_2):
            poly_4 = np.append(poly_4, poly_1[i +len(poly_2)])

    return poly_coef, subtraction


def generate_poly(nb_correction):

    poly_1 = np.array([0, 0])
    for i in range(nb_correction-1):
    
        poly_2 = np.array([0, i+1])
        res_poly = poly_multi(poly_1, poly_2, log_table)
        poly_1 = res_poly
    
    return res_poly


def error_level(err_level):
   
   match err_level:
    case "l" | "low":
        return 7, np.array([0, 1])
    
    case "m" | "medium":
        return 10, np.array([0, 0])
    
    case "q" | "quartile":
        return 13, np.array([1, 1])
    
    case "h" | "hight":
        return 17, np.array([1, 0])

def put_mask(len_qr_code, save_grid, forbidden_cell):

    # Search the best mask
    best_penalty = float("inf")
    for num_mask in range(8):
        penalty = 0
        grid = save_grid.copy()

        for i in range(len_qr_code):
            for j in range(len_qr_code):

                if not any(np.array_equal(sub, np.array([i, j]))  for sub in forbidden_cell) and mask(num_mask, j, i):
                    grid[j, i] = 1 - grid[j, i]


        #Evalute
        penalty += evalutaion_1(grid)
        penalty += evalutaion_2(grid)
        penalty += evalutaion_3(grid)
        penalty += evalutaion_4(grid)

        if penalty < best_penalty:
            best_mask = num_mask
            best_penalty = penalty

    print("best", best_mask)

    #Apply the best mask
    grid = save_grid.copy()
    for i in range(len_qr_code):
        for j in range(len_qr_code):
            if not any(np.array_equal(sub, np.array([i, j]))  for sub in forbidden_cell) and mask(best_mask, j, i):
                grid[j, i] = 1 - grid[j, i]
    
    return grid, best_mask

def format_string(grid, array_level_coor, best_mask):

    array_num_mask = np.unpackbits(np.array([best_mask], dtype=np.uint8))[5:]

    format_string = np.array([], np.byte)
    format_string = np.append(format_string, array_level_coor) 
    format_string = np.append(format_string, array_num_mask) 
    format_string = np.append(format_string, np.zeros(10)).astype(np.byte)
    save_gen_poly =  np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], np.byte)


    while len(format_string) > 11:
        format_string = np.delete(format_string, 0).astype(np.byte)

        if len(format_string) > len(save_gen_poly):
            gen_poly = np.append(save_gen_poly, np.zeros(len(format_string) - len(save_gen_poly))).astype(np.byte)

        else:
            gen_poly = save_gen_poly.astype(np.byte)

        format_string = np.bitwise_xor(gen_poly, format_string)


    format_string = np.delete(format_string, 0)

    while len(format_string) < 10:
        format_string = np.insert(format_string, 0, 0)

    arr = np.array([])
    arr = np.append(arr, array_level_coor)
    arr = np.append(arr, array_num_mask)
    format_string = np.insert(format_string, 0, arr).astype(np.byte)
    format_string = np.bitwise_xor(format_string, np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]))

    print("format string", format_string)

    grid[8, 0:6] = format_string[0:6]
    grid[8, 7:9] = format_string[6:8]
    grid[21-7:21, 8] = format_string[0:7][::-1]
    grid[7, 8] = format_string[8]

    grid[0:6, 8] = format_string[9:15][::-1]
    grid[8, 21-8:21] = format_string[7:15]

    return grid



len_qr_code = 21
message = input("Wittre something: \n")
print("Your message is ", message, "\n")

er_level = input("Which is the level error: \n")
print("Your message is ", er_level, "\n")
nb_correction, array_level_error = error_level(er_level.lower())

forbidden_cell = make_forbidden_cell(len_qr_code)

#The polynomial message is in base log 
#The generaate polynome is in base exp
message_poly =  encode(message, 26-nb_correction)
gen_poly = generate_poly(nb_correction)

error_poly = message_poly
save_gen_poly = log_table[gen_poly][:, 0]

#Division
for _ in range(26-nb_correction):
    
    #Apply the exp to the polinomial message and the gen polyninome
    error_poly = log_table[error_poly][:, 1]
    gen_poly = log_table[save_gen_poly][:, 1]
    gen_poly += error_poly[0]
    gen_poly = gen_poly%255

    #Apply the log to the polinomial message and the gen polyninome
    gen_poly = log_table[gen_poly][:, 0]
    error_poly = log_table[error_poly][:, 0]

    #Make the polinomes the same lenght
    if len(error_poly) > len(gen_poly):
        gen_poly = np.append(gen_poly, np.zeros(len(error_poly) - len(gen_poly), np.int8))

    if len(error_poly) < len(gen_poly):
        error_poly = np.append(error_poly, np.zeros(len(gen_poly) - len(error_poly), np.int8))

    error_poly = error_poly ^ gen_poly
    error_poly = error_poly[1:]

#Data store in QR code: message + redondance
message_poly = np.append(message_poly, error_poly)

final_message = np.unpackbits(message_poly.astype(np.uint8))



# Convertir directement les entiers en hexadécimal
hex_array = [format(x, '02x') for x in message_poly]
# Ajouter des espaces entre les valeurs hexadécimales
hex_value = ' '.join(hex_array)



print("\nThe polynomial message\n", message_poly)
print("The final message\n", hex_value)

#Create the grid
grid = make_QR_Code(21, final_message, forbidden_cell)
x_offset, y_offset, cell_size = dim_grid(WIDTH, HEIGHT, 21, 21, 2)

#Apply the mask
grid, best_mask = put_mask(len_qr_code, grid, forbidden_cell)

print("best mask:", best_mask)
grid = format_string(grid, array_level_error, best_mask)


WIN = pygame.display.set_mode((WIDTH, HEIGHT))
WIN.fill(WHITE)
pygame.display.set_caption("QR code")


run = True
while run:
            
    #Pygame event
    for event in pygame.event.get():
            
        #Quit pygame
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False

    # Rafraîchit l'affichage
    draw_grid(WIN, 21, 21, cell_size, x_offset, y_offset, grid)
    pygame.display.flip()

# Ferme Pygame
pygame.quit()
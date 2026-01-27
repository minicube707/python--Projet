import numpy as np
import matplotlib.pyplot as plt
import os 

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

#Expand the limits before add \n for the numpy matrix
np.set_printoptions(linewidth=150)

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

def create_finder_patern():

    ##Create a grid a 7*7 full of 1
    patern = np.ones((7, 7), dtype=int)

    #Extren black edge
    patern[0, :] = 0
    patern[6, :] = 0
    patern[:, 0] = 0
    patern[:, 6] = 0

    #Middle square
    patern[2, 2:5] = 0
    patern[3, 2:5] = 0
    patern[4, 2:5] = 0

    return patern

def add_finder_parten(grid):

    patern  = create_finder_patern()

    grid[0:7, 0:7] = patern
    grid[0:7, -7:] = patern
    grid[-7:, 0:7] = patern

    return (grid)

def add_timing_partern(grid):

    n = grid[8:-8, 6].shape[0]
    grid[8:-8, 6] = np.arange(n) % 2
    grid[6, 8:-8] = np.arange(n) % 2

    return grid

def write_pair_data(grid, data1, data2, index_x, index_y):
    
    grid[index_y, index_x] = data1
    grid[index_y, index_x - 1] = data2

    return grid

def write_up_data(grid, data, start_x, start_y, index_data, max_index):

    nb_line = 0

    while(1):

        if (start_y < max_index):
            return grid, nb_line
        
        write_pair_data(grid, data[index_data], data[index_data + 1], start_x, start_y)
        index_data += 2
        start_y -= 1
        nb_line += 1


def write_down_data(grid, data, start_x, start_y, index_data, max_index):
    
    nb_line = 0

    while(1):

        if (start_y > max_index):
            return grid, nb_line

        write_pair_data(grid, data[index_data], data[index_data + 1], start_x, start_y)
        index_data += 2
        start_y += 1
        nb_line += 1

def write_data_first(grid, data):

    nb_line_tt = 0
    start_x = grid.shape[1] - 1
    len_grid = start_x

    #Write the data to the one third right part of the QR Code
    while (start_x > len_grid - 6):

        grid, nb_line = write_up_data(grid, data, start_x, len_grid, nb_line_tt * 2, 8)
        nb_line_tt += nb_line
        start_x -= 2

        grid, nb_line = write_down_data(grid, data, start_x, 8, nb_line_tt * 2, len_grid)
        nb_line_tt += nb_line
        start_x -= 2

    return grid, nb_line_tt, start_x

def write_data_second(grid, data, start_x, nb_line_tt):

    len_grid = grid.shape[1] - 1
    
    #Write the data to the two third middle part of the QR Code
    while (start_x > 10):

        grid, nb_line = write_up_data(grid, data, start_x, len_grid, nb_line_tt * 2, 7)
        nb_line_tt += nb_line

        #Avoid to overwrite the timing patern
        grid, nb_line = write_up_data(grid, data, start_x, 5, nb_line_tt * 2, 0)
        nb_line_tt += nb_line
        start_x -= 2

        #Avoid to overwrite the timing patern
        grid, nb_line = write_down_data(grid, data, start_x, 0, nb_line_tt * 2, 5)
        nb_line_tt += nb_line
         
        grid, nb_line = write_down_data(grid, data, start_x, 7, nb_line_tt * 2, len_grid)
        nb_line_tt += nb_line
        start_x -= 2

    return grid, nb_line_tt, start_x

def write_data_last(grid, data, start_x, nb_line_tt):

    #Write the data to the last third left part of the QR Code
    while (start_x > 0):

        grid, nb_line = write_up_data(grid, data, start_x, 12, nb_line_tt * 2, 9)
        nb_line_tt += nb_line
        start_x -= 2

        #Avoid to overwrite the timing patern
        if (start_x == 6):
            start_x -= 1

        grid, nb_line = write_down_data(grid, data, start_x, 9, nb_line_tt * 2, 12)
        nb_line_tt += nb_line
        start_x -= 2

    return grid

def write_data(grid, data):

    nb_line_tt = 0
    start_x = grid.shape[1] - 1
    len_grid = start_x

    grid, nb_line_tt, start_x = write_data_first(grid, data)
    grid, nb_line_tt, start_x = write_data_second(grid, data, start_x, nb_line_tt)
    grid = write_data_last(grid, data, start_x, nb_line_tt)

    return (grid)

def num_encode(data):

    bit_message = []

    #For all the 3 digits
    for i in range(0, len(data), 3):

        #Verify if the fisrt digits begin with a 0
        num = int(data[i:i+3])

        #Avoid divide by zero encountered in log10
        if (num == 0):
            bit_message.append(format(num, "04b"))

        elif (np.log10(num) >= 2):
            bit_message.append(format(num, "010b"))
           
        elif (np.log10(num) >= 1):
            bit_message.append(format(num, "07b"))

        else:
            bit_message.append(format(num, "04b"))

    return bit_message


def alpha_encode(data):

    bit_message = []
    alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    
    #For all the pair of char
    for i in range(0, len(data), 2):

        string = data[i:i+2]

        if (len(string) == 2):
            f = alphanumeric.find(data[i])
            s = alphanumeric.find(data[i + 1])

            num = (f * 45) + s
            bit_message.append(format(num, "011b"))
        
        else:
            num = alphanumeric.find(data[i])
            bit_message.append(format(num, "06b"))
    
    return bit_message

def bit_encode(data):

    bit_message = []
    for i in range(len(data)):

        unicode = ord(data[i])
        bit_message.append(format(unicode, "08b"))

    return bit_message

def manage_data():

    numeric = "0123456789"
    alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

    #data = input("Write something: ")
    #data = "HELLO WORLD"
    data = "AB"
    print("DATA ", data)

    len_data = len(data)

    #Numeric
    if all(digit in list(numeric) for digit in list(data)):

        print("mode: numeric")
        bit_count = format(len_data, "010b")
        bit_message = num_encode(data)

        #Add Mode Indicator & Character Count Indicator 
        bit_message.insert(0, "0001" + bit_count)
        
        return bit_message, bit_count
    
    #Alphanumeric
    elif all(char in list(alphanumeric) for char in list(data)):

        print("mode: alphanumeric")
        bit_count = format(len_data, "09b")
        bit_message = alpha_encode(data)

        #Add Mode Indicator & Character Count Indicator
        bit_message.insert(0, "0010" + bit_count)

        return bit_message, bit_count
    
    #Bit
    else:
        print("mode: bit")
        bit_count = format(len_data, "08b")
        bit_message = bit_encode(data)  

        #Add Mode Indicator & Character Count Indicator
        bit_message.insert(0, "0100" + bit_count)

    return bit_message, bit_count

def add_terminaison(bit_message):

    #=> ADD check Error Correction Code Words and Block Information <=
    #Error Correction level set as Medium 2
    error_correction_level = 2
    tt_num_codeword = 16
    tt_num_codeword_bit = tt_num_codeword * 8
    error_correction_codewords_per_block = 10

    #Add Terminator
    len_bit_message = sum(len(elem) for elem in bit_message)
    add_terminator = tt_num_codeword_bit - len_bit_message
    if (add_terminator >= 4):
        bit_message.append("0000")
        len_bit_message += 4

    else:
        bit_message.append(add_terminator * "0")
        len_bit_message += add_terminator

    #Make Length a Multiple of 8
    len_bit_message = sum(len(elem) for elem in bit_message)
    add_zero = (8 - len_bit_message % 8) % 8
    bit_message.append(add_zero * "0")
    len_bit_message += add_zero
    
    #Add Pad Byte
    nb_padding = (tt_num_codeword_bit - len_bit_message) // 8
    for i in range(nb_padding):
        if (i % 2):
            bit_message.append("00010001")
        else:
            bit_message.append("11101100")
  
    return bit_message

def add_correction(bit_message):

    # Join all the bits
    bitstream = ''.join(bit_message)

    # Split into bytes
    bytes_numbers = [bitstream[i:i+8] for i in range(0, len(bitstream), 8)]

    #Check the sequence of data codeword bytes in octect and hexadecimal
    #print([int(octet, 2) for octet in bytes_numbers])
    print([hex(int(octet, 2)) for octet in bytes_numbers])

    return bytes_numbers

#Multiply 2 polynome in Galois 256 field, the input are the exponent and the output is the exponent
def multiply_polynomials(poly_1, poly_2, log_table):

    #Initialise the size of final polynome
    res_poly = np.zeros(len(poly_1) + len(poly_2) -1, np.int16)
    
    for i in range(len(poly_1)):
        expo_1 = poly_1[i]
        for j in range(len(poly_2)):
            expo_2 = poly_2[j]    

            #Add the exponent, apply the modulo 255   
            num = (expo_1 + expo_2) % 255

            #Then Xor the log of the addition
            res_poly[i+j] ^= int(log_table[int(num)][0])

    #Retrun the exponent for each coef of the polynome
    res_poly = log_table[res_poly][:, 1]

    return res_poly


def generate_poly(nb_correction, log_table):

    poly_1 = np.array([0, 0])
    for i in range(nb_correction-1):
    
        poly_2 = np.array([0, i+1])
        res_poly = multiply_polynomials(poly_1, poly_2, log_table)
        poly_1 = res_poly
    
    return res_poly

def polynomial_long_division(data_poly_exp, gene_poly_exp, log_table):

    gene_poly_exp += data_poly_exp[0]
    gene_poly_exp %= 255

    #Convert generator poly to log 
    gene_poly_log = log_table[gene_poly_exp][:, 0]

    #Convert generator poly to log 
    data_poly_log = log_table[data_poly_exp][:, 0]

    #Fill with zeros to fit the size 
    if (len(data_poly_log) > len(gene_poly_log)):
        gene_poly_log = np.append(gene_poly_log, np.zeros(len(data_poly_log) - len(gene_poly_log), dtype=int))
    else:
        data_poly_log = np.append(data_poly_log, np.zeros(len(gene_poly_log) - len(data_poly_log), dtype=int))
    
    # Apply the condition: if gene_poly_log == 0, then take data_poly_log, otherwise perform an XOR
    print("Gen ", gene_poly_log)
    print("Dat ", data_poly_log)
    gene_poly_log ^= data_poly_log
    
    #Pop the first element of the generator poly
    gene_poly_log = np.delete(gene_poly_log, 0)

    #Convert generator poly to exp 
    gene_poly_exp = log_table[gene_poly_log][:, 1]

    return (gene_poly_exp)

def manage_error_correction(bit_message):
    
    #log_table = np.load("log_table.npy")
    
    ecc_block_size = 10
    #The division has been performed 16 times, which is the number of terms in the message polynomial
    tt_num_codeword = 16

    gene_poly_exp = generate_poly(ecc_block_size, log_table)
    data_poly_log = np.array([int(octet, 2) for octet in bit_message])
    data_poly_exp = log_table[data_poly_log][:, 1]
    
    
    print("Len data ", len(data_poly_exp))
    print("Len gene ", len(gene_poly_exp))
    print("data ", data_poly_log)
    print("gene ", gene_poly_exp)
    print("")
    
    res = data_poly_exp
    for i in range(tt_num_codeword):
        res = polynomial_long_division(res, gene_poly_exp.copy(), log_table)
        print("res ", res)


    #Error Correction Code in log
    ecc = log_table[res][:, 0]

    #print(ecc)
    print("ECC ", [hex(octet) for octet in ecc])


def main():

    #Create a grid a 21*21 full of 1
    grid = np.ones((21, 21))

    grid = add_finder_parten(grid)
    grid = add_timing_partern(grid)

    data = np.arange((216))

    bit_message, bit_count = manage_data()
    bit_message = add_terminaison(bit_message)
    bit_message = add_correction(bit_message)
    manage_error_correction(bit_message)

    #print(bit_message)
    write_data(grid, data)
    
    #plt.figure()
    #plt.imshow(grid, cmap='gray')
    #plt.show()

main()
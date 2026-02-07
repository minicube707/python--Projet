import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import os 
import json
import re

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

#Expand the limits before add \n for the numpy matrix
np.set_printoptions(linewidth=150)

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                     MAKE PATTERN                        │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def create_finder_pattern():

    ##Create a grid a 7*7 full of 0
    pattern = np.zeros((7, 7), dtype=int)

    #Extren black edge
    pattern[0, :] = 1
    pattern[6, :] = 1
    pattern[:, 0] = 1
    pattern[:, 6] = 1

    #Middle square
    pattern[2, 2:5] = 1
    pattern[3, 2:5] = 1
    pattern[4, 2:5] = 1
    
    return pattern

def add_finder_partten(grid):

    pattern  = create_finder_pattern()

    #Init pattern then add finding pattern. Only the space for format information remains.
    grid[0:9, 0:9] = 0
    grid[0:9, -8:] = 0
    grid[-8:, 0:9] = 0

    #Add Finding Pattern
    grid[0:7, 0:7] = pattern
    grid[0:7, -7:] = pattern
    grid[-7:, 0:7] = pattern

    return (grid)

def create_alignment_patterns(grid):

    pattern = np.ones((5, 5), dtype=int)
    pattern[1:4, 1:4] = np.zeros((3, 3), dtype=int)
    pattern[2, 2] = 1
    
    return pattern

def add_alignment_patterns(grid, forbidden_node, int_qr_level, list_alignment_pos):

    #No Alignement Pattern for QR Code level 1
    if (int_qr_level == 1):
        return grid
    
    pattern = create_alignment_patterns(grid)

    #For each combinaison
    for combination in list_alignment_pos:

        x, y = combination[0], combination[1]
        area = {(xi, yi) for xi in range(x - 2, x + 3) for yi in range(y - 2, y + 3)}

        if not area & forbidden_node:
            grid[y-2:y+3, x-2:x+3] = pattern

    return grid
    
    
def add_timing_partern(grid):

    n = grid[8:-8, 6].shape[0]
    grid[8:-8, 6] = 1 - np.arange(n) % 2
    grid[6, 8:-8] = 1 - np.arange(n) % 2

    return grid

def add_pattern(grid, forbiden_node, int_qr_level, list_alignment_pos):

    grid = add_finder_partten(grid)
    grid = add_timing_partern(grid)
    grid = add_alignment_patterns(grid, forbiden_node, int_qr_level, list_alignment_pos)

    return (grid)


# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                    VERSION STRING                       │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def add_version_string(grid, int_qr_level):

    bit_qr_level = format(int_qr_level, "06b")
    data_poly = np.array([int(bit) for bit in bit_qr_level], dtype=np.uint8)

    #Generator Polynomial
    gene_poly = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    len_ecc_bits = 12
    ecc_format_bit = calcul_division_BCH(data_poly, gene_poly, len_ecc_bits)
    
    #Put the Format and Error Correction Bits Together
    version_string = np.append(data_poly, ecc_format_bit)

    #Bottom-left
    grid[-11:-8, 0:6] = version_string[::-1].reshape((6, 3)).T

    #Top-right
    grid[0:6, -11:-8] = version_string[::-1].reshape((6, 3))

    return grid

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                    FORMAT STRING                        │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def remove_left_zeros(lst):
    i = 0
    while i < len(lst) and lst[i] == 0:
        i += 1
    return lst[i:]

#Uses polynomial division over the binary finite field GF(2) to generate parity bits for BCH error-correcting codes, 
#designed to correct random errors
def calcul_division_BCH(data_poly, generator_poly, ecc_bits):

    #Generate Error Correction
    data_poly = np.append(data_poly, np.zeros(ecc_bits)).astype(dtype=int)

    #Remove Any 0s from the Left Side
    data_poly = remove_left_zeros(data_poly)
    
    while(len(data_poly) > ecc_bits):

        #Pad the Generator Polynomial string on the right with 0s to make it the same length as the current format string:
        pad_gen_poly = np.append(generator_poly, np.zeros(len(data_poly) - len(generator_poly))).astype(dtype=int)

        #XOR the padded generator polynomial string with the current format string.
        data_poly ^= pad_gen_poly

        #Remove any 0s from the Left Side
        data_poly = remove_left_zeros(data_poly)

    #If the result were smaller than len(ecc_bits) bits, we would pad it on the LEFT with 0s to make it len(ecc_bits) bits long.
    if (data_poly.size < ecc_bits):
        data_poly = np.append(np.zeros(ecc_bits - data_poly.size, dtype=int), data_poly)

    return (data_poly)

def put_format_string(grid, format_string):

    #Top Left Corner
    grid[8, 0:6] = format_string[0:6]
    grid[8, 7:9] = format_string[6:8]
    grid[7, 8] = format_string[8]
    grid[:6, 8] = format_string[9:][::-1]

    #Bottom Left Corner
    grid[-7:, 8] = format_string[:7][::-1]

    #Black point
    grid[-8, 8] = 1

    #Top Right Corner
    grid[8, -8:] = format_string[7:]

    return grid

def add_format_string(grid, num_mask, ecc_level):

    list_format_string = []

    #Add Error Correction Bits
    #L
    if (ecc_level == "L"):
        list_format_string.append("01")

    #M
    elif (ecc_level == "M"):
        list_format_string.append("00")
    
    #Q
    elif (ecc_level == "Q"):
        list_format_string.append("11")

    #H
    else:
        list_format_string.append("10")

    #Add Number Mask Pattern
    list_format_string.append(format(num_mask, "03b"))

    #Convertion list string to list numpy
    data_poly = np.array([int(bit) for byte in list_format_string for bit in byte], dtype=np.uint8)

    #Generator Polynomial
    gene_poly = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=int)
    len_ecc_bits = 10
    ecc_format_bit = calcul_division_BCH(data_poly, gene_poly, len_ecc_bits)

    #Put the Format and Error Correction Bits Together
    format_string = np.append(data_poly, ecc_format_bit)

    # Mask String 101010000010010
    CONST_MASK_STRING = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])

    #XOR with the Mask String
    final_format_string = format_string ^ CONST_MASK_STRING

    #Put format string on QR CODE
    grid = put_format_string(grid, final_format_string)

    return grid

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │              CREATE FORBIDEN NODE                       │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def add_forbiden_node_finding_pattern(forbidden, N):

    #Top Left
    for y in range(0, 9):
        for x in range(0, 9):
            forbidden.add((x, y))

    #Top Right
    for y in range(0, 9):
        for x in range(N - 8, N):
            forbidden.add((x, y))

    #Bottom Left
    for y in range(N - 8, N):
        for x in range(0, 9):
            forbidden.add((x, y))

    return forbidden

def add_forbiden_node_timing_pattern(forbidden, N):

    for i in range(8, N - 8):
        #Horizontal
        forbidden.add((i, 6))

        #Vertical
        forbidden.add((6, i))

    return forbidden

def create_forbiden_node_alignement_pattern(forbidden, list_value):
    
    #For each combinaison
    for combination in list_value:
        center_x, center_y = combination[0], combination[1]

        #Create the forbidden node
        set_pattern = set()
        for y in range(center_y - 2, center_y + 3):
            for x in range(center_x - 2, center_x + 3):
                set_pattern.add((x, y))

        #If `set_pattern` has no coordinates in common with `forbidden`,
        #add all coordinates from `set_pattern` to `forbidden`
        if not set_pattern & forbidden:
            forbidden = forbidden.union(set_pattern)

    return forbidden

def add_forbiden_node_alignement_pattern(forbidden, N, int_qr_level, list_alignment_pos):

    #No Alignement Pattern for QR Code level 1
    if (int_qr_level == 1):
        return forbidden
    
    forbidden = create_forbiden_node_alignement_pattern(forbidden, list_alignment_pos)
    return forbidden

def add_forbiden_node_versoin_pattern(forbidden, N, int_qr_level):

    if (1 <= int_qr_level <= 6):
        return forbidden
    
    #Bottom Left
    for y in range(N - 11, N - 8):
        for x in range(0, 6):
            forbidden.add((x, y))

    for x in range(N - 11, N - 8):
        for y in range(0, 6):
            forbidden.add((x, y))

    return forbidden

def create_list_forbiden_node(int_qr_level, list_alignment_pos):

    N = qr_dimension(int_qr_level)
    forbidden = set()

    forbidden = add_forbiden_node_finding_pattern(forbidden, N)
    forbidden = add_forbiden_node_alignement_pattern(forbidden, N, int_qr_level, list_alignment_pos)
    forbidden = add_forbiden_node_timing_pattern(forbidden, N)
    forbidden = add_forbiden_node_versoin_pattern(forbidden, N, int_qr_level)
    
    return forbidden

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                     WRITE DATA                          │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def write_pair_data(grid, data1, data2, index_x, index_y, forbiden_node):

    nb_write = 0

    if (index_y, index_x) not in forbiden_node:
        grid[index_y, index_x] = data1
        nb_write += 1

    if (index_y, index_x - 1) not in forbiden_node:
        grid[index_y, index_x - 1] = data2
        nb_write += 1

    return grid, nb_write

def write_up_data(grid, data, start_x, index_data, forbiden_node):

    nb_write = 0
    index_x = start_x
    index_y = grid.shape[1] - 1

    while(1):

        if (index_y < 0):
            return grid, nb_write
        
        if (index_y, index_x) not in forbiden_node:
            grid[index_y, index_x] = data[index_data + nb_write]
            nb_write += 1

        if (index_y, index_x - 1) not in forbiden_node:
            grid[index_y, index_x - 1] = data[index_data + nb_write]
            nb_write += 1

        index_y -= 1

def write_down_data(grid, data, start_x, index_data, forbiden_node):

    nb_write = 0
    index_x = start_x
    index_y = 0

    while(1):

        if (index_y > grid.shape[1] - 1):
            return grid, nb_write
        
        if (index_x, index_y) not in forbiden_node:
            grid[index_y, index_x] = data[index_data + nb_write]
            nb_write += 1

        if (index_x - 1, index_y) not in forbiden_node:
            grid[index_y, index_x - 1] = data[index_data + nb_write]
            nb_write += 1

        index_y += 1

def write_data_first(grid, data, forbiden_node):

    nb_line_tt = 0
    start_x = grid.shape[1] - 1

    #Write the data to the first two third middle part of the QR Code
    while (start_x > 10):

        grid, nb_write = write_up_data(grid, data , start_x, nb_line_tt, forbiden_node)
        nb_line_tt += nb_write
        start_x -= 2

        grid, nb_write = write_down_data(grid, data, start_x, nb_line_tt, forbiden_node)
        nb_line_tt += nb_write
        start_x -= 2

    return grid, nb_line_tt, start_x

def write_data_last(grid, data, start_x, nb_line_tt,  forbiden_node):

    #Write the data to the last third left part of the QR Code
    while (start_x > 0):
        
        grid, nb_write =  write_up_data(grid, data, start_x, nb_line_tt, forbiden_node)
        nb_line_tt += nb_write
        start_x -= 2

        #Avoid to overwrite the timing pattern
        if (start_x == 6):
            start_x -= 1

        grid, nb_write =  write_down_data(grid, data, start_x, nb_line_tt, forbiden_node)
        nb_line_tt += nb_write
        start_x -= 2

    return grid

def qr_dimension(version):
    return 21 + 4 * (version - 1)

def write_data(data, int_qr_level, list_alignment_pos):

    forbiden_node = create_list_forbiden_node(int_qr_level, list_alignment_pos)

    dim = qr_dimension(int_qr_level)
    grid = np.zeros((dim, dim))

    nb_line_tt = 0
    start_x = dim - 1

    grid, nb_line_tt, start_x = write_data_first(grid, data, forbiden_node)
    grid = write_data_last(grid, data, start_x, nb_line_tt,  forbiden_node)

    return grid

def add_remainder_bit(bit_message, int_qr_level):

    #If the QR Code level 1.
    #Add 0 0s
    if (int_qr_level == 1):
        return bit_message
    
    #If the QR Code level is between 2 and 6.
    #Add 7 0s
    if (2 <= int_qr_level <= 6):
        bit_message.append(7 * "0")
        return bit_message

    #If the QR Code level is between 7 and 13.
    #Add 0 0s
    if (7 <= int_qr_level <= 13):
        return bit_message
    
    #If the QR Code level is between 14 and 20.
    #Add 3 0s
    if (14 <= int_qr_level <= 20):
        bit_message.append(3 * "0")
        return bit_message
    
    #If the QR Code level is between 21 and 27.
    #Add 4 0s
    if (21 <= int_qr_level <= 27):
        bit_message.append(4 * "0")
        return bit_message
    
    #If the QR Code level is between 28 and 34.
    #Add 3 0s
    if (28 <= int_qr_level <= 34):
        bit_message.append(3 * "0")
        return bit_message
    
    #If the QR Code level is between 35 and 40.
    #Add 0 0s
    if (35 <= int_qr_level <= 40):
        return bit_message
    
# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                  GET INPUT USER                         │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def check_input():

    correct_ecc_level = False
    correct_qr_level =  False

    while(not correct_ecc_level or not correct_qr_level):
        
        if (not correct_ecc_level):
            ecc_level = input("\nWhich level of Encryption do you want ?:\n")
            print("Input ", ecc_level)

        if (not correct_qr_level):
            qr_level = input("\nWhich level of QR Code do you want ?:\n")
            print("Input ", qr_level)

        if (ecc_level not in {"L", "M", "Q", "H", "0"} or len(ecc_level) != 1):
            print("\nWrong input for Encode.")
            print("Please write: 'L', 'M', 'Q', 'H' or '0' (auto-mode)")
        
        else:
            correct_ecc_level = True

        #Try cast string to int
        try:
            int_qr_level = int(qr_level)

        #If cast fail
        except ValueError:
            print("\nWrong input for the Level of QR Code.")
            print("Error: input is not a digit")

        #Else check if the number is between 1 and 40
        else:

            if not 0 <= int_qr_level <= 40:
                print("\nWrong input for the Level of QR Code.")
                print("Please write a digit between 0 and 40, 0 is auto-mode")

            else:
                correct_qr_level = True

    return ecc_level, int_qr_level

def ecc_qr_level_defined(rs, ecc_level, int_qr_level, type_data, len_data):

    #Check if the data isn't longer than the lenght max for a error code correction
    max_len = rs["versions"][str(int_qr_level)]["capacities"][ecc_level][type_data]
    if (len_data > max_len):
        print("\nOverflow Len data")
        exit()

    print("\nLevel Error Code Correction: ", ecc_level)
    print("Level of the QR Code: ", int_qr_level)
    return  ecc_level, int_qr_level

def ecc_level_undefined(rs, int_qr_level, type_data, len_data):

    #Mode auto
    list_ecc_level =  ["H", "Q", "M", "L"]

    for i in range(4):

        if (rs["versions"][str(int_qr_level)]["capacities"][list_ecc_level[i]][type_data] > len_data):
            print("\nLevel Error Code Correction: ", list_ecc_level[i])
            print("Level of the QR Code: ", int_qr_level)
            return list_ecc_level[i], int_qr_level
    
    print("\nOverflow Len data")
    exit()

def qr_level_undefined(rs, ecc_level, type_data, len_data):

    #Mode auto
    for i in range(1, 41):

        if (rs["versions"][str(i)]["capacities"][ecc_level][type_data] > len_data):
            print("\nLevel Error Code Correction: ", ecc_level)
            print("Level of the QR Code: ", str(i))
            return ecc_level, i
    
    print("\nOverflow Len data")
    exit()

def ecc_qr_level_undefined(rs, type_data, len_data):

    list_ecc_level =  ["H", "Q", "M", "L"]

    for i in range(1, 41):
        for j in range(4):

            if (rs["versions"][str(i)]["capacities"][list_ecc_level[j]][type_data] > len_data):
                print("\nLevel Error Code Correction: ", list_ecc_level[j])
                print("Level of the QR Code: ", str(i))
                return list_ecc_level[j], i
    
    print("\nOverflow Len data")
    exit()

def get_ecc_level(type_data, len_data):

    ecc_level, int_qr_level = check_input()
    
    with open("utils/qr_capacity.json", "r") as f:
        rs = json.load(f)

    if ecc_level != "0" and int_qr_level != 0:
        return (ecc_qr_level_defined(rs, ecc_level, int_qr_level, type_data, len_data))
       
    if ecc_level == "0" and int_qr_level != 0:
        return (ecc_level_undefined(rs, int_qr_level, type_data, len_data))
    
    if ecc_level != "0" and int_qr_level == 0:
        return (qr_level_undefined(rs, ecc_level, type_data, len_data))

    else:
        return (ecc_qr_level_undefined(rs, type_data, len_data))
    
# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │          DATA ANALYSIS & DATA ENCODING                  │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def num_encode(data):

    bit_message = []

    #For all the 3 digits
    for i in range(0, len(data), 3):
        chunk = data[i:i+3]
        num = int(chunk)

        if len(chunk) == 3:
            bit_message.append(format(num, "010b"))

        elif len(chunk) == 2:
            bit_message.append(format(num, "07b"))

        else:  # len == 1
            bit_message.append(format(num, "04b"))

    return bit_message


def alpha_encode(data):

    bit_message = []
    ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    
    #For all the pair of char
    for i in range(0, len(data), 2):

        string = data[i:i+2]

        if (len(string) == 2):
            f = ALPHANUMERIC.find(data[i])
            s = ALPHANUMERIC.find(data[i + 1])

            num = (f * 45) + s
            bit_message.append(format(num, "011b"))
        
        else:
            num = ALPHANUMERIC.find(data[i])
            bit_message.append(format(num, "06b"))
    
    return bit_message

def byte_encode(data):
    
    #Cast the string in bytes UTF-8
    byte_data = data.encode('utf-8')

    #Cast each bytes in bit
    bit_message = [format(b, '08b') for b in byte_data]
    
    return bit_message

def manage_data():

    NUMERIC = "0123456789"
    ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

    data = input("Write something: ")
    print("\nData: ", data)

    #Numeric
    if all(digit in list(NUMERIC) for digit in list(data)):

        bit_message = num_encode(data)
        type_data = "numeric"
        
    #Alphanumeric
    elif all(char in list(ALPHANUMERIC) for char in list(data)):

        bit_message = alpha_encode(data)
        type_data = "alphanumeric"

    #Bit
    else:
        bit_message = byte_encode(data)
        type_data = "byte"

    return bit_message, data, type_data

#Add Mode Indicator & Character Count Indicator 
def add_mi_cci(bit_message, data, type_data, int_qr_level):

    if type_data == "numeric":
        bit_count = format(len(bit_message), "010b")
        bit_message.insert(0, "0001" + bit_count)
        return bit_message

    if type_data == "alphanumeric":
        bit_count = format(len(bit_message), "09b")
        bit_message.insert(0, "0010" + bit_count)
        return bit_message
    
    if type_data == "byte":
        if int_qr_level < 10 and len(data) < 256:
            bit_count = format(len(bit_message), "08b")
        else:
            bit_count = format(len(bit_message), "016b")
        bit_message.insert(0, "0100" + bit_count)
        
        return bit_message
    
def add_terminaison(bit_message, tt_num_codeword):

    tt_num_codeword_bit = tt_num_codeword * 8

    #Add Terminator
    #If it Remains Enough Space Add Terminator
    len_bit_message = sum(len(elem) for elem in bit_message)
    add_terminator = tt_num_codeword_bit - len_bit_message
    if (add_terminator >= 4):
        bit_message.append("0000")
        len_bit_message += 4

    #Else complete with 0
    else:
        bit_message.append(add_terminator * "0")
        len_bit_message += add_terminator

    #Make Length a Multiple of 8
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

def convert_to_bytes(bit_message):

    # Join all the bits
    bitstream = ''.join(bit_message)

    # Split into bytes
    bytes_numbers = [bitstream[i:i+8] for i in range(0, len(bitstream), 8)]

    return bytes_numbers

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                ERROR CODE CORRECTOR                     │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

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
            res_poly[i+j] ^= int(log_table[num][0])

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

    #If the First Element is a log(0). In Tab Log, log(0) == -1. Shift the Polynomial
    if (data_poly_exp[0] == -1):

        #Pop the First Eement of the Data Polynomial
        data_poly_exp = np.delete(data_poly_exp, 0)

        return data_poly_exp

    #Multiply the Generator Polynomial by the First Term of the Data Polynomial
    gene_poly_exp += data_poly_exp[0]

    #Apply modulo to stay in Galois Field (256)
    gene_poly_exp %= 255

    #Convert Generator Polynomial to Log
    gene_poly_log = log_table[gene_poly_exp][:, 0]

    #Convert Data Polynomial to Log
    #EXEPT for the log (0) stay 0
    #In tab log, log(0) == -1
    data_poly_log = np.where(data_poly_exp == -1, 0, log_table[data_poly_exp][:, 0])

    #Fill with zeros to fit the size 
    if (len(data_poly_log) > len(gene_poly_log)):
        gene_poly_log = np.append(gene_poly_log, np.zeros(len(data_poly_log) - len(gene_poly_log), dtype=int))

    else:
        data_poly_log = np.append(data_poly_log, np.zeros(len(gene_poly_log) - len(data_poly_log), dtype=int))

    #Apply XOR to the Data Polynomial
    data_poly_log ^= gene_poly_log

    #Pop the First Element of the Data Polynomial
    data_poly_log = np.delete(data_poly_log, 0)

    #Convert Data Polynomial to Exp 
    data_poly_exp = log_table[data_poly_log][:, 1]

    return (data_poly_exp)

def generate_reed_solomon_ecc(bit_message, data_codewords_per_block, ecc_block_size):
    
    log_table = np.load("utils/log_table.npy")

    #Encryption Reed Solomon
    gene_poly_exp = generate_poly(ecc_block_size, log_table)
    data_poly_log = np.array([int(octet, 2) for octet in bit_message])
    data_poly_exp = log_table[data_poly_log][:, 1]

    #The division has been performed total number of codewords, which is the number of terms in the message polynomial
    res = data_poly_exp
    for i in range(data_codewords_per_block):
        res = polynomial_long_division(res, gene_poly_exp.copy(), log_table)

    #Convert Error Correction Code to log
    #EXEPT for the log (0) stay 0
    #In tab log, log(0) == -1
    ecc = np.where(res == -1, 0, log_table[res][:, 0])

    #If len ECC is shorter than len ECC Block Size. Complete with 0 at the end
    if (ecc.size < ecc_block_size):
        ecc = np.append(ecc, np.zeros(ecc_block_size - len(ecc))).astype(dtype=int)

    list_ecc = [format(x, "08b") for x in ecc]
    return (list_ecc)

def manage_error_correction(bit_message, dict_ecc_info):

    #Get Data form the Dict
    ecc_block_size = dict_ecc_info["ec_codewords_per_block"]
    list_block_info = dict_ecc_info["block_info"]
    
    list_ecc = []
    list_data_block = []

    #If there is 2 groups
    if (len(list_block_info) == 2):
        size_block_data_group1 = list_block_info[0]["data_codewords_per_block"]
    
    #If there is only one group
    else:
        size_block_data_group1 = 0

    #For each block
    for k in range(len(list_block_info)):
        
        #Get the info from the block
        num_blocks = list_block_info[k]["num_blocks"]
        data_codewords_per_block = list_block_info[k]["data_codewords_per_block"]

        #Create list of Data Block
        for i in range(num_blocks):
            list_data_block.append(bit_message[(data_codewords_per_block * i) + (k * 2 * size_block_data_group1): data_codewords_per_block * (i + 1) + (k * 2 *size_block_data_group1)])

        #Create list of Error Code Corrector
        for i in range(num_blocks):
            list_ecc.append(generate_reed_solomon_ecc(list_data_block[i + k * len(list_block_info)], data_codewords_per_block, ecc_block_size))
    
    #Create the bit message
    bit_message = []
    for j in range(data_codewords_per_block):
        for k in range(len(list_block_info)):
            for i in range(num_blocks):
                
                #Check if we are not in out of range
                if (j < len(list_data_block[i + k * len(list_block_info)])):
                    bit_message.append(list_data_block[i + k * len(list_block_info)][j])

    #Add the ECC
    for j in range(ecc_block_size):
        for k in range(len(list_block_info)):
            for i in range(num_blocks):

                bit_message.append(list_ecc[i + k * len(list_block_info)][j])

    return bit_message

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                    PENALTY PATTERNS                     │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def penalty_run(line):
    penalty = 0
    current = line[0]
    length = 1

    for v in line[1:]:
        if v == current:
            length += 1
        else:
            if length >= 5:
                penalty += 3 + (length - 5)
            current = v
            length = 1

    #Last run
    if length >= 5:
        penalty += 3 + (length - 5)

    return penalty

def penalty_1(grid):
    penalty = 0

    #Horizontal
    for row in grid:
        penalty += penalty_run(row)

    #Vertical
    for col in grid.T:
        penalty += penalty_run(col)

    return penalty

def penalty_2(grid):

    #Window slice 2*2
    blocks = sliding_window_view(grid, (2, 2))

    #Test bloc all 0 or all 1
    zeros = np.all(blocks == 0, axis=(2, 3))
    ones  = np.all(blocks == 1, axis=(2, 3))

    return np.sum(zeros | ones) * 3

def penalty_3(grid):

    count = 0

    PATTERN = np.array([1,0,1,1,1,0,1])
    WHITE_4 = np.array([0,0,0,0])

    #Horizontal
    for row in grid:
        windows = sliding_window_view(row, 11)
        for w in windows:
            if (np.array_equal(w[:7], PATTERN) and np.array_equal(w[7:], WHITE_4)) or \
               (np.array_equal(w[:4], WHITE_4) and np.array_equal(w[4:], PATTERN)):
                count += 1

    #Vertical
    for col in grid.T:
        windows = sliding_window_view(col, 11)
        for w in windows:
            if (np.array_equal(w[:7], PATTERN) and np.array_equal(w[7:], WHITE_4)) or \
               (np.array_equal(w[:4], WHITE_4) and np.array_equal(w[4:], PATTERN)):
                count += 1

    return count * 40

def penalty_4(grid):

    #Count how many dark modules there are in the matrix
    nb_dark = np.count_nonzero(grid == 1)

    #Calculate the percent of modules in the matrix that are dark: (darkmodules / totalmodules) * 100
    pourcent_dark = nb_dark / grid.size * 100

    #Determine the previous and next multiple of five of this percent
    previous_m5 = (pourcent_dark // 5) * 5
    next_m5 = previous_m5 + 5

    #Subtract 50 from each of these multiples of five and take the absolute value of the result
    abs_p = abs(50 - previous_m5)
    abs_n = abs(50 - next_m5)
    
    #Divide each of these by five
    p = abs_p // 5
    n = abs_n // 5

    #Finally, take the smallest of the two numbers and multiply it by 10
    penalty = min(p, n) * 10

    return penalty

def calcul_penalty(grid):

    penalty = penalty_1(grid)
    penalty += penalty_2(grid)
    penalty += penalty_3(grid)
    penalty += penalty_4(grid)

    return penalty

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                    DATA MASKING                         │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>        MASK          >>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>

#(row + column) mod 2 == 0
def apply_mask_0(grid, int_qr_level):

    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (i+j) % 2 == 0
    mask = (i + j) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#(row) mod 2 == 0
def apply_mask_1(grid, int_qr_level):

    # Create index i, j
    i, _ = np.indices(grid.shape)

    # Mask where i % 2 == 0
    mask = i % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#(column) mod 3 == 0
def apply_mask_2(grid, int_qr_level):

    # Create index i, j
    _, j = np.indices(grid.shape)

    # Mask where i % 3 == 0
    mask = j  % 3 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#(row + column) mod 3 == 0
def apply_mask_3(grid, int_qr_level):

    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (i+j) % 3 == 0
    mask = (i + j) % 3 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#( floor(row / 2) + floor(column / 3) ) mod 2 == 0
def apply_mask_4(grid, int_qr_level):

    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (i // 2 + j // 3) % 2 == 0
    mask = (i // 2 + j // 3) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#((row * column) mod 2) + ((row * column) mod 3) == 0
def apply_mask_5(grid, int_qr_level):


    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where ((i * j) % 2 + (i * j) % 3 ) == 0
    mask = ((i * j) % 2 + (i * j) % 3 ) == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#( ((row * column) mod 2) + ((row * column) mod 3) ) mod 2 == 0
def apply_mask_6(grid, int_qr_level):


    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (((i * j) % 2 + (i * j) % 3 ) % 2) == 0
    mask = ((i * j) % 2 + (i * j) % 3 ) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

#( ((row + column) mod 2) + ((row * column) mod 3) ) mod 2 == 0
def apply_mask_7(grid, int_qr_level):


    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (((i + j) % 2 + (i * j) % 3 ) % 2) == 0
    mask = ((i + j) % 2 + (i * j) % 3 ) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    return grid

def data_masking(grid, ecc_level, int_qr_level, list_alignment_pos):

    list_mask_function = [apply_mask_0, apply_mask_1, apply_mask_2, apply_mask_3, apply_mask_4, apply_mask_5, apply_mask_6, apply_mask_7]

    best_mask = 0
    best_penalty = 10000

    N = qr_dimension(int_qr_level)
    forbidden = set()
    forbidden_node = add_forbiden_node_finding_pattern(forbidden, N)

    for i, f in enumerate(list_mask_function):
        cp_grid = f(grid.copy(), int_qr_level)
        cp_grid = add_pattern(cp_grid, forbidden_node, int_qr_level, list_alignment_pos)
        cp_grid = add_format_string(cp_grid, i, ecc_level)
        cp_grid = add_version_string(cp_grid, int_qr_level)

        penalty = calcul_penalty(cp_grid)

        if (penalty < best_penalty):
            best_penalty = penalty
            best_mask = i

    #Return the QR Code with the best penalty
    cp_grid = list_mask_function[best_mask](grid, int_qr_level)
    cp_grid = add_pattern(cp_grid, forbidden_node, int_qr_level, list_alignment_pos)
    cp_grid = add_format_string(cp_grid, best_mask, ecc_level)
    cp_grid = add_version_string(cp_grid, int_qr_level)
    print("Mask: ", best_mask)

    return cp_grid

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │              GET DATA FROM JSON FILE                    │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def import_data_json_file(ecc_level, int_qr_level):

    sub_folder = "utils/"

    with open(sub_folder + "qr_rs_structure.json", "r") as f:
        rs = json.load(f)

    ecc_info = rs["versions"][str(int_qr_level)]["error_correction"][ecc_level]

    with open(sub_folder +  "qr_alignment_patterns.json", "r") as f:
        alignment_positions = json.load(f)

    if int_qr_level != 1:
        positions = alignment_positions[str(int_qr_level)]
        list_alignment_pos = [(x, y) for x in positions for y in positions]

    else:
        list_alignment_pos = []

    return list_alignment_pos, {
        "total_data_codewords": ecc_info["total_data_codewords"],
        "ec_codewords_per_block": ecc_info["ec_codewords_per_block"],
        "block_info": ecc_info["blocks"]
    }

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                          MAIN                           │
# │                                                         │   
# └─────────────────────────────────────────────────────────┘

def main():

    bit_message, data, type_data = manage_data()
    ecc_level, int_qr_level = get_ecc_level(type_data, len(bit_message))
    list_alignment_pos, dict_ecc_info = import_data_json_file(ecc_level, int_qr_level)
    bit_message = add_mi_cci(bit_message, data, type_data, int_qr_level)
    bit_message = add_terminaison(bit_message, dict_ecc_info["total_data_codewords"])
    bit_message = convert_to_bytes(bit_message)
    bit_message = manage_error_correction(bit_message, dict_ecc_info)
    bit_message = add_remainder_bit(bit_message, int_qr_level)

    #Convertion list string to list numpy
    list_numpy = np.array([int(bit) for byte in bit_message for bit in byte], dtype=np.uint8)

    grid = write_data(list_numpy, int_qr_level, list_alignment_pos)
    grid = data_masking(grid, ecc_level, int_qr_level, list_alignment_pos)

    #Change char that cannot be use to save the QR Code
    safe_data = re.sub(r'[^a-zA-Z0-9_-]', '_', data)
    
    #Trunc if too long
    max_filename_len = 50
    if len(safe_data) > max_filename_len:
        safe_data = safe_data[:max_filename_len]

    plt.figure()
    plt.title(data)
    plt.imshow(1 - grid, cmap='gray')
    plt.axis("off")
    plt.savefig("export/" + safe_data + ".png")
    plt.show()

main()
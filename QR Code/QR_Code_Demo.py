import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import os 
import json

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

#Expand the limits before add \n for the numpy matrix
np.set_printoptions(linewidth=150)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>        UTILS         >>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>

def print_bit(res):
    
    for i in range(len(res)):
        print(res[i], end="")
    print("")

def print_flag(name):
    width = 27  # longueur de la ligne de "="
    print("\n" + "=" * width)
    print(name.center(width))
    print("=" * width)

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                     MAKE PATTERN                        │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def create_finder_pattern():

    ##Create a grid a 7*7 full of 0
    patern = np.zeros((7, 7), dtype=int)

    #Extren black edge
    patern[0, :] = 1
    patern[6, :] = 1
    patern[:, 0] = 1
    patern[:, 6] = 1

    #Middle square
    patern[2, 2:5] = 1
    patern[3, 2:5] = 1
    patern[4, 2:5] = 1
    
    return patern

def add_finder_partten(grid):

    patern  = create_finder_pattern()

    #Init pattern then add finding pattern. Only the space for format information remains.
    grid[0:9, 0:9] = 0
    grid[0:9, -8:] = 0
    grid[-8:, 0:9] = 0

    #Add Finding Pattern
    grid[0:7, 0:7] = patern
    grid[0:7, -7:] = patern
    grid[-7:, 0:7] = patern

    return (grid)

def add_timing_partern(grid):

    n = grid[8:-8, 6].shape[0]
    grid[8:-8, 6] = 1 - np.arange(n) % 2
    grid[6, 8:-8] = 1 - np.arange(n) % 2

    return grid

def add_pattern(grid):

    grid = add_finder_partten(grid)
    grid = add_timing_partern(grid)

    return (grid)

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

def calcul_error_correction_bit(data_poly):

    #Generator Polynomial
    gene_poly = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=int)

    #Generate Error Correction
    data_poly = np.append(data_poly, np.zeros(10)).astype(dtype=int)

    #Remove Any 0s from the Left Side
    data_poly = remove_left_zeros(data_poly)
    
    while(len(data_poly) > 10):

        #Pad the Generator Polynomial string on the right with 0s to make it the same length as the current format string:
        pad_gen_poly = np.append(gene_poly, np.zeros(len(data_poly) - len(gene_poly))).astype(dtype=int)

        #XOR the padded generator polynomial string with the current format string.
        data_poly ^= pad_gen_poly

        #Remove any 0s from the Left Side
        data_poly = remove_left_zeros(data_poly)

    #If the result were smaller than 10 bits, we would pad it on the LEFT with 0s to make it 10 bits long.
    if (data_poly.size < 10):
        data_poly = np.append(np.zeros(10 - data_poly.size, dtype=int), data_poly)

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

    ecc_format_bit = calcul_error_correction_bit(data_poly)

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
# │                     WRITE DATA                          │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

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

        grid, nb_line = write_up_data(grid, data, start_x, len_grid, nb_line_tt * 2, 9)
        nb_line_tt += nb_line
        start_x -= 2

        grid, nb_line = write_down_data(grid, data, start_x, 9, nb_line_tt * 2, len_grid)
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

def write_data(data):

    #Create a grid a 21*21 full of 0
    grid = np.zeros((21, 21))

    nb_line_tt = 0
    start_x = grid.shape[1] - 1

    grid, nb_line_tt, start_x = write_data_first(grid, data)
    grid, nb_line_tt, start_x = write_data_second(grid, data, start_x, nb_line_tt)
    grid = write_data_last(grid, data, start_x, nb_line_tt)

    return (grid)

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │          DATA ANALYSIS & DATA ENCODING                  │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def get_ecc_level(mode, len_data):

    while(1):

        ecc_level = input("Which level of Encryption do you want ?:\n")
        print("Input ", ecc_level)

        if (ecc_level not in "LMQH0"):
            print("Wrong input.")
            print("Please write: 'L', 'M', 'Q', 'H' or '0' (auto-mode)")
        
        else:
            break
    
    with open("utils/qr_capacity.json", "r") as f:
        rs = json.load(f)

    if ecc_level != "0":
        
        #Check if the data isn't longer than the lenght max for a error code correction
        max_len = rs["capacities"][ecc_level][mode]
        if (len_data > max_len):
            print("Overflow Len data")
            exit()

        print("Level Error Code Correction: ", ecc_level)
        return  ecc_level
    
    #Mode auto
    list_ecc_level =  ["H", "Q", "M", "L"]

    for i in range(4):

        if (rs["capacities"][list_ecc_level[i]][mode] > len_data):
            print("Level Error Code Correction: ", list_ecc_level[i])
            return list_ecc_level[i]
    
    print("Overflow Len data")
    exit()
    
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

    print("Encode Data:", bit_message)
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
    
    print("Encode Data:", bit_message)
    return bit_message

def bit_encode(data):

    bit_message = []
    for i in range(len(data)):

        unicode = ord(data[i])
        bit_message.append(format(unicode, "08b"))

    print("Encode Data:", bit_message)
    return bit_message

def manage_data():

    numeric = "0123456789"
    alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

    data = input("Write something: ")

    print("Data ", data)
    print("")

    len_data = len(data)

    #Numeric
    if all(digit in list(numeric) for digit in list(data)):

        print("mode: numeric")
        bit_count = format(len_data, "010b")
        bit_message = num_encode(data)
        mode = "numeric"

        #Add Mode Indicator & Character Count Indicator 
        bit_message.insert(0, "0001" + bit_count)
        
    #Alphanumeric
    elif all(char in list(alphanumeric) for char in list(data)):

        print("mode: alphanumeric")
        bit_count = format(len_data, "09b")
        bit_message = alpha_encode(data)
        mode = "alphanumeric"

        #Add Mode Indicator & Character Count Indicator
        bit_message.insert(0, "0010" + bit_count)

    #Bit
    else:
        print("mode: bit")
        bit_count = format(len_data, "08b")
        bit_message = bit_encode(data)  
        mode = "byte"

        #Add Mode Indicator & Character Count Indicator
        bit_message.insert(0, "0100" + bit_count)

    print("Mode Indicator: ", bit_message[0][:4])
    print("Character Count Indicator: ", bit_message)
    return bit_message, data, mode

def add_terminaison(bit_message, tt_num_codeword):

    tt_num_codeword_bit = tt_num_codeword * 8

    #Add Terminator
    #If it Remains Enough Space Add Terminator
    len_bit_message = sum(len(elem) for elem in bit_message)
    print("\nTotal len bit: ", len_bit_message)

    add_terminator = tt_num_codeword_bit - len_bit_message
    if (add_terminator >= 4):
        bit_message.append("0000")
        len_bit_message += 4

        print("There is more than 4 bit remaining")
        print("Add 0000")

    #Else complete with 0
    else:
        bit_message.append(add_terminator * "0")
        len_bit_message += add_terminator

        print("There is less than 4 bit remaining")
        print("Add ", add_terminator * "0")

    #Make Length a Multiple of 8
    print("\nTotal len bit: ", len_bit_message)

    add_zero = (8 - len_bit_message % 8) % 8
    bit_message.append(add_zero * "0")
    len_bit_message += add_zero
    
    print("\nMake the Lenght a Multiplr of 8")
    print("Add ", add_zero * "0")

    #Add Pad Byte
    nb_padding = (tt_num_codeword_bit - len_bit_message) // 8

    print("\nComplete the remaning bytes with an alternation of 00010001 and 11101100")
    print("Bytes remaining: ", nb_padding)

    for i in range(nb_padding):
        if (i % 2):
            bit_message.append("00010001")
            print("Add: ", "00010001")

        else:
            bit_message.append("11101100")
            print("Add: ", "11101100")
  
    return bit_message

def add_correction(bit_message):

    # Join all the bits
    bitstream = ''.join(bit_message)

    # Split into bytes
    bytes_numbers = [bitstream[i:i+8] for i in range(0, len(bitstream), 8)]

    #Check the sequence of data codeword bytes in octect and hexadecimal
    print("\nFinal Data Encoding")
    print("Bit: ", bytes_numbers)
    print("Octal: ", [int(octet, 2) for octet in bytes_numbers])
    print("Hexa: ", [hex(int(octet, 2)) for octet in bytes_numbers])

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

    print("\n=========")
    print("Polynomial Long Division")
    print("=========")

    print("\nPolynomial Data with Exposant: ", data_poly_exp)
    print("Polynomial Generator with Exposant: ", gene_poly_exp)

    #Multiply the generator polynome by the first term of the data polynome
    gene_poly_exp += data_poly_exp[0]
    print("\nAdd the First Element of the Data Polynomial to the Generator Polynomial")
    print("Polynomial Generator with Exposant: ", gene_poly_exp)
    
    #Apply modulo to stay in Galois Field (256)
    gene_poly_exp %= 255
    print("\nApply Modulo 255 to the Generator Polynomial")
    print("Polynomial Generator with Exposant: ", gene_poly_exp)

    #Convert generator poly to log
    #EXEPT for the log (0) stay 0
    #In tab log, log(0) == -1
    gene_poly_log = np.where(gene_poly_exp == -1, 0, log_table[gene_poly_exp][:, 0])
    print("\nConvert the Generator Polynomial to the Log Base")
    print("Polynomial Generator with Log: ", gene_poly_log)

    #Convert data poly to log
    #EXEPT for the log (0) stay 0
    #In tab log, log(0) == -1
    data_poly_log = np.where(data_poly_exp == -1, 0, log_table[data_poly_exp][:, 0])
    print("\nConvert the Data Polynomial to the Log Base")
    print("Polynomial Data with Log: ", data_poly_log)

    #Fill with zeros to fit the size 
    print("\nFill the Polynomail to have the same lenght")
    if (len(data_poly_log) > len(gene_poly_log)):
        print("Add ", (len(data_poly_log) - len(gene_poly_log)) * "0", " to the Polynomial Generator with Log")
        gene_poly_log = np.append(gene_poly_log, np.zeros(len(data_poly_log) - len(gene_poly_log), dtype=int))
        print("Polynomial Generator with Log: ", gene_poly_log)

    else:
        print("Add ", (len(gene_poly_log) - len(data_poly_log)) * "0", " to the Polynomial Data with Log")
        data_poly_log = np.append(data_poly_log, np.zeros(len(gene_poly_log) - len(data_poly_log), dtype=int))
        print("Polynomial Data with Log: ", data_poly_log)

    #Apply XOR
    gene_poly_log ^= data_poly_log
    print("\nApply XOR to the Polynomial Generator with Polynomial Data")
    print("Polynomial Generator with Log: ", gene_poly_log)

    #Pop the first element of the generator poly
    gene_poly_log = np.delete(gene_poly_log, 0)
    print("Remove the first element to the Polynomial Generator with Log")

    #Convert generator poly to exp 
    gene_poly_exp = log_table[gene_poly_log][:, 1]
    print("\nConvert the Generator Polynomial to the log base")
    print("Polynomial Generator with Exp: ", gene_poly_exp)

    return (gene_poly_exp)

def manage_error_correction(bit_message, tt_num_codeword, ecc_block_size):
    
    log_table = np.load("utils/log_table.npy")

    #Encryption Reed Solomon
    gene_poly_exp = generate_poly(ecc_block_size, log_table)

    print("\nCreate a Generator Polynomial for ", ecc_block_size, "size")
    print("Polynomial Generator with Exposant: ", gene_poly_exp)

    data_poly_log = np.array([int(octet, 2) for octet in bit_message])
    data_poly_exp = log_table[data_poly_log][:, 1]

    print("Use the Log Table to Express the Data Polynomial in Exposant Form")
    print("Polynomial Data with Exposant: ", data_poly_exp)

    #The division has been performed 16 times, which is the number of terms in the message polynomial
    print("\nApply the Polynomial Long Division ", tt_num_codeword, " time")

    res = data_poly_exp
    for i in range(tt_num_codeword):
        res = polynomial_long_division(res, gene_poly_exp.copy(), log_table)

    print("\n=========")
    print("Error Code Corrector")
    print("=========")

    print("\nThe Polynomial Remaining is the Error Code Corrector")
    print("Error Code Corrector with Exp: ", res)

    #Convert Error Correction Code to log
    #EXEPT for the log (0) stay 0
    #In tab log, log(0) == -1
    ecc = np.where(res == -1, 0, log_table[res][:, 0])
    print("\nConvert theError Code Corrector to the log base")
    print("Error Code Corrector with Log: ", ecc)

    print("\nBit: ", [format(x, "08b") for x in ecc])
    print("Octal: ", [oct(octet) for octet in ecc])
    print("Hexa:  ", [hex(octet) for octet in ecc])

    bit_message.extend(format(x, "08b") for x in ecc)

    return (bit_message)

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

    print("Calcul Penalty")
    penalty = penalty_1(grid)
    penalty += penalty_2(grid)
    penalty += penalty_3(grid)
    penalty += penalty_4(grid)

    print("Penalty N°1: ", penalty_1(grid))
    print("Penalty N°2: ", penalty_2(grid))
    print("Penalty N°3: ", penalty_3(grid))
    print("Penalty N°4: ", penalty_4(grid))
    print("Sum Penalty: ", penalty)

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
def apply_mask_0(grid):

    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (i+j) % 2 == 0
    mask = (i + j) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#(row) mod 2 == 0
def apply_mask_1(grid):

    # Create index i, j
    i, _ = np.indices(grid.shape)

    # Mask where i % 2 == 0
    mask = i % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#(column) mod 3 == 0
def apply_mask_2(grid):

    # Create index i, j
    _, j = np.indices(grid.shape)

    # Mask where i % 3 == 0
    mask = j  % 3 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#(row + column) mod 3 == 0
def apply_mask_3(grid):

    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (i+j) % 3 == 0
    mask = (i + j) % 3 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#( floor(row / 2) + floor(column / 3) ) mod 2 == 0
def apply_mask_4(grid):

    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (i // 2 + j // 3) % 2 == 0
    mask = (i // 2 + j // 3) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#((row * column) mod 2) + ((row * column) mod 3) == 0
def apply_mask_5(grid):


    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where ((i * j) % 2 + (i * j) % 3 ) == 0
    mask = ((i * j) % 2 + (i * j) % 3 ) == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#( ((row * column) mod 2) + ((row * column) mod 3) ) mod 2 == 0
def apply_mask_6(grid):


    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (((i * j) % 2 + (i * j) % 3 ) % 2) == 0
    mask = ((i * j) % 2 + (i * j) % 3 ) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

#( ((row + column) mod 2) + ((row * column) mod 3) ) mod 2 == 0
def apply_mask_7(grid):


    # Create index i, j
    i, j = np.indices(grid.shape)

    # Mask where (((i + j) % 2 + (i * j) % 3 ) % 2) == 0
    mask = ((i + j) % 2 + (i * j) % 3 ) % 2 == 0

    # Flip bits where mask is true
    grid[mask] = 1 - grid[mask]

    #Add pattern
    grid = add_pattern(grid)

    return grid

def data_masking(grid, ecc_level):

    list_mask_function = [apply_mask_0, apply_mask_1, apply_mask_2, apply_mask_3, apply_mask_4, apply_mask_5, apply_mask_6, apply_mask_7]

    best_mask = 0
    best_penalty = 10000

    for i, f in enumerate(list_mask_function):
        print("\nEval Mask N°", i)

        cp_grid = f(grid.copy())
        cp_grid = add_pattern(cp_grid)
        cp_grid = add_format_string(cp_grid, i, ecc_level)

        penalty = calcul_penalty(cp_grid)

        if (penalty < best_penalty):
            best_penalty = penalty
            best_mask = i
    
    print("\nBest Mask is ", best_mask, " Mask with ", best_penalty, " score")

    #Return the QR Code with the best penalty
    cp_grid = list_mask_function[best_mask](grid)
    cp_grid = add_pattern(cp_grid)
    cp_grid = add_format_string(cp_grid, best_mask, ecc_level)

    return cp_grid

def get_rs_structure(ecc_level):

    with open("utils/qr_rs_structure.json", "r") as f:
        rs = json.load(f)
    
    tt_num_codeword = rs["error_correction"][ecc_level]["total_data_codewords"]
    ecc_block_size = rs["error_correction"][ecc_level]["ec_codewords_per_block"]

    print("\nTotal Data Codewords: ", tt_num_codeword)
    print("Error Code Correction per Block: ", ecc_block_size)

    return tt_num_codeword, ecc_block_size

# ┌─────────────────────────────────────────────────────────┐
# │                                                         │
# │                          MAIN                           │
# │                                                         │
# └─────────────────────────────────────────────────────────┘

def main():

    print_flag("Encode Data")
    bit_message, data, mode = manage_data()
    ecc_level = get_ecc_level(mode, len(data))
    tt_num_codeword, ecc_block_size = get_rs_structure(ecc_level)
    bit_message = add_terminaison(bit_message, tt_num_codeword)
    bit_message = add_correction(bit_message)

    print_flag("DATA ANALYSIS & DATA ENCODING")
    bit_message = manage_error_correction(bit_message, tt_num_codeword, ecc_block_size)
    
    print("\nFinal Codewords")
    print("Bit: ", bit_message)
    print("Octal: ", [oct(int(octet, 2)) for octet in bit_message])
    print("Hexa:  ", [hex(int(octet, 2)) for octet in bit_message])
 
    #Convertion list string to list numpy
    list_numpy = np.array([int(bit) for byte in bit_message for bit in byte], dtype=np.uint8)
    
    grid = write_data(list_numpy)

    print_flag("DATA MASKING")
    grid = data_masking(grid, ecc_level)
    
    plt.figure()
    plt.title(data)
    plt.imshow(1 - grid, cmap='gray')
    plt.axis("off")
    plt.savefig("export/" + data + ".png")
    plt.show()

main()
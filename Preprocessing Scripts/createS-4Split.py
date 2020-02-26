import csv
import sys
import warnings

import numpy as np


def get_number_of_words(txt_file):
    with open(txt_file) as file:
        return sum(1 for _ in file)


def get_number_of_columns(csv_file):
    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        return len(next(reader))


def create_alphabet_dictionary(csv_file):
    alphabet_dict = dict()

    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)

        for index, line in enumerate(reader):
            alphabet_dict[line[0]] = index

    return alphabet_dict


def write_s_file(write_file, matrix, words):
    with open(write_file, "w+") as file:
        for row_number, row in enumerate(matrix):
            file.write(words[row_number] + "," + ",".join(np.char.mod('%f', row)) + "\n")


def half_string(string):
    print(string)
    half_1 = string[: int(len(string) / 2)]
    half_2 = string[int(len(string) / 2):]
    return half_1, half_2


def quarter_string(string):
    print(string)
    first_half, second_half = half_string(string)
    print(first_half)

    one, two = half_string(first_half)
    three, four = half_string(second_half)

    return [one, two, three, four]


def __main__():
    if len(sys.argv) < 4:
        exit("Not enough arguments given, needs alphabet csv, label/word txt and output s matrix")

    alphabet_csv = sys.argv[1]
    word_txt = sys.argv[2]
    s_matrix_csv = sys.argv[3]

    number_of_words = get_number_of_words(word_txt)
    alphabet_dict = create_alphabet_dictionary(alphabet_csv)
    csv_num_cols = get_number_of_columns(alphabet_csv)
    numpy_csv = np.genfromtxt(alphabet_csv, dtype=float, delimiter=",", filling_values=1)
    # First column is letter count, then 4 times the features
    s_matrix = np.zeros((number_of_words, csv_num_cols * 4 - 3))

    word_list = []

    with open(word_txt, "r") as file:
        for word_index, line in enumerate(file):
            split_line = line.split(maxsplit=1)
            class_index = split_line[0]
            word = split_line[1].rstrip()
            word_list.append(word)
            numpy_word = None

            word_parts = quarter_string(word)

            for part_index, part in enumerate(word_parts):
                # 0,12 - 12, 23 - 23, 34 - 34,45
                numpy_start = 1
                if part_index == 0:
                    start_column = 1
                    end_column = csv_num_cols
                    numpy_start = 0
                elif part_index == 1:
                    start_column = csv_num_cols
                    end_column = (csv_num_cols * 2) - 1
                elif part_index == 2:
                    start_column = (csv_num_cols * 2) - 1
                    end_column = (csv_num_cols * 3) - 2
                elif part_index == 3:
                    start_column = (csv_num_cols * 3) - 2
                    end_column = (csv_num_cols * 4) - 3

                print(start_column, end_column)

                for letter in part:
                    if letter is '\n':
                        continue

                    try:
                        letter_index = alphabet_dict[letter]
                        print(len(s_matrix[word_index][start_column:end_column]),
                              len(numpy_csv[letter_index][numpy_start:]))
                        s_matrix[word_index][start_column:end_column] += numpy_csv[letter_index][1:]
                        s_matrix[word_index][0] += 1
                    except:
                        warnings.warn("Key '%s' not found in dictionary" % letter)

            divider = s_matrix[word_index][0]
            s_matrix[word_index][0] = 1 / divider

            for col_index in range(1, csv_num_cols * 4 - 3):
                s_matrix[word_index][col_index] = s_matrix[word_index][col_index] / divider

    write_s_file(s_matrix_csv, s_matrix, word_list)

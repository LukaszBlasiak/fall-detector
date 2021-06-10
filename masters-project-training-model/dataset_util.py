import os
from os import listdir
import numpy as np
from os.path import isfile, join
import random

dirname = os.path.dirname(__file__)
dataset_path = os.path.join(dirname, 'saved_datasets\\8x8x9')

# amount of (history) frames to keep in buffer
max_buffer_size = 8


def load_dataset(test_split_ratio, shuffle=False):
    x_train = []
    y_train = []
    dataset_files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
    if shuffle:
        random.shuffle(dataset_files)
    i = 0
    while i < len(dataset_files):
        pixels_from_file = load_pixels_from_file(os.path.join(dataset_path, dataset_files[i]))
        data_class = get_class_from_filename(dataset_files[i])
        x_train.append(pixels_from_file)
        y_train.append([data_class])
        i += 1
    test_split_ratio = int(len(x_train) * test_split_ratio)
    x_test, x_train = np.split(np.asarray(x_train), [test_split_ratio])
    y_test, y_train = np.split(np.asarray(y_train), [test_split_ratio])
    return x_train, x_test, y_train, y_test


def load_pixels_from_file(file_path):
    frames = []
    try:
        fp = open(file_path, 'r')
        i = 0
        while i < max_buffer_size:
            line = fp.readline()
            row = line.split(";")
            row.pop()
            image_vector = np.array(row, dtype=float)
            frames.append(image_vector)
            i += 1
    except Exception as e:
        print(str(e))
    finally:
        fp.close()
    flat_list = [item for sublist in frames for item in sublist]
    return flat_list


def get_class_from_filename(file_name):
    class_index = find_nth(file_name, "_", 2) + 1
    class_as_string = file_name[class_index]
    return int(class_as_string)


def find_nth(text, char, n):
    start = text.find(char)
    while start >= 0 and n > 1:
        start = text.find(char, start+len(char))
        n -= 1
    return start

import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import serial
import time
import signal
from scipy.ndimage import interpolation
from tkinter import *
from threading import Lock
import threading

matplotlib.use('TkAgg')
frame_size = 8
max_buffer_size = 8


def denoise_frame_entry_point(frame):
    denoised_frame = denoise_frame(frame)
    denoised_frame = denoise_frame(denoised_frame)
    denoised_frame = boost_frame(denoised_frame)
    return denoised_frame


def denoise_frame(frame):
    new_frame = np.zeros((8, 8))
    for j in range(frame_size):  # columns
        for i in range(frame_size):  # rows
            new_frame[i][j] = find_min(frame, i, j)
    return new_frame


def find_min(frame, row, col):
    minimum = frame[row][col]
    neighbours = []
    if col > 0:
        neighbours.append(frame[row][col - 1])
    if col < frame_size - 1:
        neighbours.append(frame[row][col + 1])
    if row > 0:
        neighbours.append(frame[row - 1][col])
    if row < frame_size - 1:
        neighbours.append(frame[row + 1][col])
    for i in neighbours:
        if i < minimum and (minimum - i) <= 0.5:
            minimum = i
    if np.count_nonzero(np.asarray(neighbours) == minimum) < 3 or (np.asarray(neighbours) >= frame[row][col]).sum() > 0:
        return frame[row][col]
    return minimum - 0.5


def boost_frame(frame):
    avg = np.average(frame)
    new_frame = np.zeros((8, 8))
    for j in range(frame_size):  # columns
        for i in range(frame_size):  # rows
            new_frame[i][j] = (frame[i][j] / (avg - 0.8)) * frame[i][j]
    return new_frame


def get_entropy(buffered_frames):
    entropy_array = np.zeros((8, 8))
    i = 2
    while i < max_buffer_size:
        tmp_array = np.subtract(buffered_frames[i - 2], buffered_frames[i])
        tmp_array = np.absolute(tmp_array)
        tmp_array[tmp_array <= 0.75] = 0
        entropy_array = np.add(entropy_array, tmp_array)
        i += 2
    return np.sum(entropy_array, dtype=np.float32)

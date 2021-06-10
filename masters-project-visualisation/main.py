import os
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
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
from frame_util import denoise_frame_entry_point, get_entropy

# size of subplots (bigger resolutions allows to visualize more history frames)
figure_rows = 2
figure_columns = 4

# amount of (history) frames to keep in buffer
max_buffer_size = 2 * 4

# frame resolution (eg. 8 means 8x8), should be divisible by 8
frame_size = 16

# arduino configuration (for debugging purposes on PC - in production frames are obtained directly from camera)
port = 'COM3'
baud_rate = 250000

# relative dir path to same dataset
save_path = os.path.join(os.getcwd(), 'saved_datasets')

mutex = Lock()


def exit_handler(sig, frame):
    fig.clf()
    exit()


def initialize_buffered_images(initial_item):
    interpolated_initial_item = interpolation.zoom(initial_item, frame_size / 8)
    i = 0
    while i < max_buffer_size:
        if i < figure_rows * figure_columns:
            ax.append(fig.add_subplot(figure_rows, figure_columns, i + 1))
            ax[i].set_axis_off()
        buffered_images.append(plt.imshow(interpolated_initial_item, vmin=20, vmax=30, interpolation='nearest'))
        buffered_images[i].set_cmap('hot')
        i += 1


def initialize_buffered_frames(initial_item):
    i = 0
    while i < max_buffer_size:
        buffered_frames.append(initial_item)
        i += 1


# insert new frame to buffer
def shift_buffered_frames(current_frame):
    mutex.acquire()
    try:
        i = max_buffer_size-1
        while i > 0:
            buffered_frames[i] = buffered_frames[i-1]
            i -= 1
        buffered_frames[0] = current_frame
    finally:
        mutex.release()


# refresh images according to buffered frames
def refresh_images():
    i = 0
    while i < max_buffer_size:
        interpolated_frame = interpolation.zoom(buffered_frames[i], frame_size / 8)
        buffered_images[i].set_data(interpolated_frame)
        i += 1


# save frames to output file
def save_frames():
    # 1 = fall, 0 = non fall
    current_class = 1
    mutex.acquire()
    file = None
    try:
        date_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        filename = '8x8_' + str(max_buffer_size) + '_' + str(current_class) + ' - ' + str(date_time) + '.txt'
        path_to_save = os.path.join(save_path, filename)
        file = open(path_to_save, 'w+')
        file.close()
        f = open(path_to_save, 'ab')
        i = 0
        while i < max_buffer_size:
            savetxt(f, buffered_frames[i].flatten(), newline=';', fmt='%1.2f')
            i += 1
            f.write("\n".encode())
        f.write(str(current_class).encode())
        f.close()
        time.sleep(2)
    except Exception as e:
        print(str(e))
        file.close()
    finally:
        mutex.release()


# start drawing subplots with heat maps
def start_recording_with_visualisation():
    signal.signal(signal.SIGINT, exit_handler)
    ser = serial.Serial(port, baud_rate)
    time.sleep(2)

    frame_vector = np.zeros((8, 8))
    initialize_buffered_frames(frame_vector)
    initialize_buffered_images(frame_vector)
    ser.flushInput()
    ser.flushOutput()
    time.sleep(.1)
    current = ser.read_until(terminator=b'\n').decode("ascii")
    plt.colorbar()
    while True:
        try:
            current = ser.read_until(terminator=b'\n').decode("ascii")
            frame_vector = np.array(current.split(";"), dtype=float)
            frame_vector.shape = (8, 8)
            # frame_vector = interpolation.zoom(frame_vector, frame_size / 8)
            frame_vector = denoise_frame_entry_point(frame_vector)
            print(get_entropy(buffered_frames))
            shift_buffered_frames(frame_vector)
            refresh_images()
        except Exception as e:
            print(str(e))
        finally:
            plt.draw(), plt.pause(0.000001)


# start tkinter gui render (with save button etc.)
def start_gui_render():
    top = Tk()
    top.geometry("150x150")
    button = Button(top, text="save frames", command=save_frames, height=150, width=150)
    button.pack()
    top.mainloop()


buffered_frames = []
buffered_images = []
ax = []
fig = plt.figure(figsize=(6, 4), dpi=70)

# create separate thread to render subplots and read frames from thermal camera
thread = threading.Thread(target=start_gui_render, args=())
thread.daemon = True
thread.start()

start_recording_with_visualisation()










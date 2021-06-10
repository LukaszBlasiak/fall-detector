import base64
import json
import keyboard
import matplotlib
import os
import requests

import matplotlib.pyplot as plt
import numpy as np
import serial
import time
from scipy.ndimage import interpolation
from threading import Lock

from svm import classify_frames
# CHANGE ABOVE IMPORT IN ORDER TO USE DIFFERENT CLASSIFIER!

matplotlib.use('TkAgg')

DEVICE_UUID = 'a5bb21fe-05e2-4ff2-8de6-3f571a656d6a'
PASSWORD = 'haslo123'
PATIENT_PESEL = ''

API_BASE_URL = 'https://rvnsqkh3qd.execute-api.us-east-1.amazonaws.com/prod'
DEVICE_URL = '/devices/{uuid}'
SAVE_DETECTION_URL = '/patients/{pesel}/detections'

# disables GPU support, comment this out in order to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# size of subplots (bigger resolutions allows to visualize more history frames)
figure_rows = 2
figure_columns = 4

# amount of (history) frames to keep in buffer
max_buffer_size = 8

# frame resolution (eg. 8 means 8x8), should be divisible by 8
frame_size = 8

# arduino configuration (for debugging purposes on PC - in production frames are obtained directly from camera)
port = 'COM3'
baud_rate = 250000
ser = serial.Serial(port, baud_rate)
time.sleep(2)

mutex = Lock()


# initialize buffered images with first N initial plot images (where N is size of history buffer)
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


# initialize buffered images with first N initial frames (where N is size of history buffer)
def initialize_buffered_frames(initial_item):
    i = 0
    while i < max_buffer_size:
        buffered_frames.append(initial_item)
        i += 1


# insert new frame to buffer and shift buffer
def shift_buffered_frames(current_frame):
    mutex.acquire()
    try:
        i = max_buffer_size - 1
        while i > 0:
            buffered_frames[i] = buffered_frames[i - 1]
            i -= 1
        buffered_frames[0] = current_frame
    finally:
        mutex.release()


# refresh plot images according to stored frames in buffer
def refresh_images():
    i = 0
    while i < max_buffer_size:
        interpolated_frame = interpolation.zoom(buffered_frames[i], frame_size / 8)
        buffered_images[i].set_data(interpolated_frame)
        i += 1


# start capture images from thermal camera and draw then on subplots
def start_recording_with_visualisation():
    frame_vector = np.zeros((8, 8))
    initialize_buffered_frames(frame_vector)
    initialize_buffered_images(frame_vector)
    ser.flushInput()
    ser.flushOutput()
    time.sleep(.1)
    # start reading from camera in order to synchronize connection
    current = ser.read_until(terminator=b'\n').decode("ascii")

    while True:
        capture_frame()
        flatted_buffer = get_flatted_buffer()
        prediction = classify_frames(flatted_buffer)
        if prediction == 1:
            print('Fall suspicion!')
            verify_detection(flatted_buffer)


def capture_frame():
    try:
        current = ser.read_until(terminator=b'\n').decode("ascii")
        frame_vector = np.array(current.split(";"), dtype=float)
        frame_vector.shape = (8, 8)
        frame_vector = interpolation.zoom(frame_vector, frame_size / 8)
        shift_buffered_frames(frame_vector)
        refresh_images()
    except Exception as e:  # work on python 3.x
        print(str(e))
    finally:
        plt.draw(), plt.pause(0.000001)


# converts 2d frames buffer (creating local copy instead of modifying original one) into 1d array and returns it
def get_flatted_buffer():
    flat_buffer = []
    i = 0
    # flatmap each frame
    while i < max_buffer_size:
        flat_frame = [item for sublist in buffered_frames[i] for item in sublist]
        flat_buffer.append(np.asarray(flat_frame, dtype=float))
        i += 1
    # reverse order (for visualisation purposes first element is the latest frame)
    return [item for sublist in flat_buffer for item in sublist]


# checks whether fall is reliable or just false positive. For given period of time it analise camera image for any
# violent movement using simplified entropy.
def verify_detection(flatted_buffer):
    # skip few frames to avoid self entropy generation
    i = 8
    while i > 0:
        capture_frame()
        i -= 1
    i = len(buffered_frames) - 1
    while i > 0:
        buffered_frames[i] = buffered_frames[0]
        i -= 1
    timeout = time.time() + 10
    while True:
        capture_frame()
        entropy = get_entropy()
        if entropy > 18 or keyboard.is_pressed('space'):
            print('Fall canceled')
            ser.flushInput()
            return
        if time.time() > timeout:
            print('Fall confirmed')
            save_detection_record_in_db(flatted_buffer)
            ser.flushInput()
            return


def get_entropy():
    entropy_array = np.zeros((8, 8))
    i = 2
    while i < max_buffer_size:
        tmp_array = np.subtract(buffered_frames[i - 2], buffered_frames[i])
        tmp_array = np.absolute(tmp_array)
        tmp_array[tmp_array <= 0.75] = 0
        entropy_array = np.add(entropy_array, tmp_array)
        i += 2
    return np.sum(entropy_array, dtype=np.float32)


def save_detection_record_in_db(flatted_buffer):
    header = DEVICE_UUID + ":" + PASSWORD
    # prepare authorization header in following format in base64: pesel:password
    header_bytes = header.encode('ascii')
    encoded_header = base64.b64encode(header_bytes).decode('ascii')
    auth_header = {'basic_auth': encoded_header}

    frames_as_string = ";".join(map(str, flatted_buffer))
    data = json.dumps({'frames': frames_as_string})

    actual_save_detection_url = SAVE_DETECTION_URL.replace('{pesel}', PATIENT_PESEL)
    response = requests.post(API_BASE_URL + actual_save_detection_url, headers=auth_header, data=data)
    if response.status_code == 200:
        print("Saved successfully")
    else:
        print("An unexpected occurred while saving data to AWS DB")


def get_patient_pesel():
    global PATIENT_PESEL
    header = DEVICE_UUID + ":" + PASSWORD
    # prepare authorization header in following format in base64: pesel:password
    header_bytes = header.encode('ascii')
    encoded_header = base64.b64encode(header_bytes).decode('ascii')
    auth_header = {'basic_auth': encoded_header}

    actual_devices_url = DEVICE_URL.replace('{uuid}', DEVICE_UUID)
    response = requests.get(API_BASE_URL + actual_devices_url, headers=auth_header)
    device = response.json()
    PATIENT_PESEL = device['patient_pesel']


buffered_frames = []
buffered_images = []
ax = []
fig = plt.figure(figsize=(6, 4), dpi=70)

get_patient_pesel()
start_recording_with_visualisation()

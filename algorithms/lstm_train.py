import time
import os
import numpy as np
import cv2
import keras
from collections import deque

from algorithms.lstm_model import model, BATCH_SIZE, EPOCHS, FRAME_SIZE

video = cv2.VideoCapture('data/train.mp4')
speed = deque(np.loadtxt('data/train.txt', delimiter='\n'))

def labeled_frame():
    _, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return (speed.popleft(), frame)

buffer_length = model.get_input_shape_at(0)[1]
frame_buffer=deque(maxlen=buffer_length)
timestamp = time.strftime('%c')
os.mkdir('model/lstm/{}'.format(timestamp))
os.mkdir('logs/lstm/{}'.format(timestamp))
save_callback = keras.callbacks.ModelCheckpoint(
    'model/lstm/' + timestamp + '/weights-{val_mean_squared_error:.2f}.hdf5',
    monitor='val_mean_squared_error', verbose=0,
    save_best_only=True, save_weights_only=False,
    mode='auto', period=1
)
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs/lstm/{}'.format(timestamp),
    batch_size=BATCH_SIZE,
    update_freq='epoch'
)

training_labels=[]
training_frames=[]

for i in range(buffer_length - 1):
    frame_buffer.append(labeled_frame()[1])

while(video.isOpened()):
    try:
        label, frame = labeled_frame()
    except:
        break
    training_labels.append([label])
    frame_buffer.append(frame)
    training_frames.append(np.array(frame_buffer).reshape(FRAME_SIZE, 240,320, 1))
video.release()
training_frames = np.array(training_frames)
training_labels = np.array(training_labels)
model.fit(training_frames, training_labels, validation_split=0.25, epochs=EPOCHS, callbacks=[save_callback, tensorboard_callback])

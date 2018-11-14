import time
import os
import numpy as np
import cv2
import keras
from collections import deque

from algorithms.cnn_model import model, BATCH_SIZE, EPOCHS

video = cv2.VideoCapture('data/train.mp4')
speed = deque(np.loadtxt('data/train.txt', delimiter='\n'))

def labeled_frame():
    _, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return (speed.popleft(), frame)

buffer_length = model.get_input_shape_at(0)[3]
frame_buffer=deque(maxlen=buffer_length)
timestamp = time.strftime('%c')
os.mkdir('model/{}'.format(timestamp))
os.mkdir('logs/{}'.format(timestamp))
save_callback = keras.callbacks.ModelCheckpoint(
    'model/' + timestamp + '/weights-{val_mean_squared_error:.2f}.hdf5',
    monitor='val_mean_squared_error', verbose=0,
    save_best_only=True, save_weights_only=False,
    mode='auto', period=1
)
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(timestamp),
    batch_size=BATCH_SIZE,
    update_freq='epoch'
)

training_labels=[]
training_frames=[]

for i in range(buffer_length):
    frame_buffer.append(labeled_frame()[1])

while(video.isOpened()):
    try:
        label, frame = labeled_frame()
    except:
        break
    training_labels.append([label])
    frame_buffer.append(frame)
    training_frames.append(np.stack(frame_buffer, axis=2))

video.release()
training_frames = np.array(training_frames)
training_labels = np.array(training_labels)
print 'Read {} samples'.format(training_labels.shape[0])
model.fit(training_frames, training_labels, validation_split=0.25, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[save_callback, tensorboard_callback])

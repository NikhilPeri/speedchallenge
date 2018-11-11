import time
import numpy as np
import cv2
from collections import deque

from algorithms.model import cnn_model, BATCH_SIZE, EPOCHS


video = cv2.VideoCapture('data/train.mp4')
speed = deque(np.loadtxt('data/train.txt', delimiter='\n'))

def labeled_frame():
    _, frame = video.read()
    return (speed.popleft(), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

buffer_length = cnn_model.get_input_shape_at(0)[3]
frame_buffer=deque(maxlen=buffer_length)
# Populate Buffer
for i in range(buffer_length):
    frame_buffer.append(labeled_frame()[1])

training_frames=[]
training_labels=[]
cnn_model.load_weights('model/weights.hd5')
while(video.isOpened()):
    label, frame = labeled_frame()
    training_labels.append(label)
    frame_buffer.append(frame)
    training_frames.append(np.stack(frame_buffer, axis=2))

    if len(training_frames) == BATCH_SIZE:
        frames = np.array(training_frames)
        labels = np.array(training_labels)
        cnn_model.evaluate(frames, labels)
        history = cnn_model.fit(frames, labels, batch_size=BATCH_SIZE, epochs=EPOCHS))
        cnn_model.save_weights('model/weights.hd5')

        training_frames=[]
        training_labels=[]

video.release()

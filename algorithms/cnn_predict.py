import os
import sys
import numpy as np
import cv2
import keras
from collections import deque

from algorithms.cnn_model import model
video = cv2.VideoCapture(sys.argv[1])
model.load_weights(sys.argv[2])

def frame():
    _, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame

buffer_length = model.get_input_shape_at(0)[3]
frame_buffer=deque(maxlen=buffer_length)

training_frames=[]
for i in range(buffer_length):
    frame_buffer.append(frame())

while(video.isOpened()):
    try:
        frame_buffer.append(frame())
        training_frames.append(np.stack(frame_buffer, axis=2))
    except:
        break

video.release()

training_frames = np.array(training_frames)
predictions = model.predict(training_frames)
np.savetxt("{}.txt".format(sys.argv[1]), np.append(predictions[0][0], predictions))

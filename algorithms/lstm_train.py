import time
import numpy as np
import cv2
from collections import deque

from algorithms.lstm_model import model, BATCH_SIZE, EPOCHS

video = cv2.VideoCapture('data/train.mp4')
speed = deque(np.loadtxt('data/train.txt', delimiter='\n'))
def labeled_frame():
    _, frame = video.read()
    return (speed.popleft(), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
buffer_length = model.get_input_shape_at(0)[1]
frame_buffer=deque(maxlen=buffer_length)
# Populate Buffer
for i in range(buffer_length):
    frame_buffer.append(labeled_frame()[1])
training_frames=[]
training_labels=[]
try:
    model.load_weights('model/weights.hd5')
except:
    print "could not load previous weights"
while(video.isOpened()):
    label, frame = labeled_frame()
    training_labels.append([label])
    frame_buffer.append(frame)
    training_frames.append(np.stack(frame_buffer))

    if len(training_frames) == 2:
        frames = np.array(training_frames).reshape(2, buffer_length, 480, 640, 1)
        labels = np.array(training_labels)
        model.fit(frames, labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
        model.save_weights('model/weights.hd5')
        training_frames=[]
        training_labels=[]
video.release()

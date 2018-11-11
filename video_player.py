import numpy as np
import cv2

from collections import deque

video = cv2.VideoCapture('data/train.mp4')
speed = deque(np.loadtxt('data/train.txt', delimiter='\n'))

def labeled_frame():
    _, frame = video.read()
    return (speed.popleft(), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

cv2.namedWindow('Current Frame')

while(video.isOpened()):
    label, frame = labeled_frame()
    cv2.putText(
        frame,
        'Speed:{}'.format(label),
        (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Current Frame', frame)
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()

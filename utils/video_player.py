import cv2
import numpy as np

if __name__ == '__main__':
    video = cv2.VideoCapture('data/comma_ai/test.mp4')
    labels_1 = np.loadtxt('data/comma_ai/test.txt', delimiter='\n')
    labels_2 = np.loadtxt('data/comma_ai/test-postprocessed.txt', delimiter='\n')


    while video.isOpened():
        frame = video.read()[1]

        frame = cv2.putText(
            frame,
            str(np.round(labels_1[int(video.get(cv2.CAP_PROP_POS_FRAMES))-1], 3)),
            (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        frame = cv2.putText(
            frame,
            str(np.round(labels_2[int(video.get(cv2.CAP_PROP_POS_FRAMES))-1], 3)),
            (0, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow('video', frame)
        cv2.waitKey(27)

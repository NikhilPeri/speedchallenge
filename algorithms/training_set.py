import time
import numpy as np
import pandas as pd
import cv2


video = cv2.VideoCapture('data/train.mp4')
speed = pd.read_csv('data/train.txt', header=None, names=['speed'])
speed['frame'] = speed.index
import pdb; pdb.set_trace()

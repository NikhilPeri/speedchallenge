import os
import cv2
import numpy as np
import numpy.ma as ma
import dask.array as da
from dask.cache import Cache
from progress.bar import Bar

def visualize_optical_flow(flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = np.tile(255, (flow.shape[0], flow.shape[1]))
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dask_player(video, segmentation, optical_flow):
    Cache(2e6).register()    # Turn cache on globally
    video = cv2.VideoCapture(video)
    segmentation = da.from_npy_stack(segmentation)
    optical_flow = da.from_npy_stack(optical_flow)

    bar = Bar('Frame', max=video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.read()
    while video.isOpened():
        video_frame = video.read()[1]
        optical_flow_frame = optical_flow[bar.index].compute()
        segmentation_frame = segmentation[bar.index].compute()
        cv2.imshow('video', video_frame)
        cv2.imshow('segmentation', segmentation_frame)
        cv2.imshow('optical_flow', visualize_optical_flow(optical_flow_frame))

        cv2.waitKey(27)
        bar.next()

if __name__ == '__main__':
    dask_player('data/comma_ai/train.mp4', 'data/comma_ai/best_train_segments', 'data/comma_ai/train_optical_flow')

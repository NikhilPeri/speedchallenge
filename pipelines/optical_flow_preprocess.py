import os
import cv2
import numpy as np
import pickle
from dask import array as da
from progress.bar import Bar

def compute_optical_flow(input, output, chunksize=500):
    video = cv2.VideoCapture(input)
    bar = Bar('Processed Frame', max=video.get(cv2.CAP_PROP_FRAME_COUNT))
    chunks = []
    chunk_data = []

    prev_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
    bar.next()
    while video.isOpened():
        try:
            current_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_RGB2GRAY)
        except:
            break
        bar.next()
        chunk_data.append(cv2.calcOpticalFlowFarneback(
                prev_frame, current_frame,
                None, 0.5, 3, 20, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        ))

        if len(chunk_data) == chunksize:
            with open(os.path.join(output, '{}.npy'.format(len(chunks))), 'w+') as f:
                np.save(f, np.array(chunk_data))
                chunks.append(len(chunk_data))
                chunk_data = []

    # Save Partial Chunk
    with open(os.path.join(output, '{}.npy'.format(len(chunks))), 'w+') as f:
        np.save(f, np.array(chunk_data))
        chunks.append(len(chunk_data))

    with open(os.path.join(output, 'info'), 'w+') as f:
        pickle.dump({
            'chunks': (tuple(chunks), (chunk_data[0].shape[0],), (chunk_data[0].shape[1],), (chunk_data[0].shape[2],)),
            'dtype': chunk_data[0].dtype,
            'axis': 0
        }, f)

    video.release()
    print 'Split {} into {} chunks'.format(input, len(chunks))

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    with dask.config.set(pool=ThreadPool(1)):
        #compute_optical_flow('data/comma_ai/train.mp4', 'data/comma_ai/train_optical_flow')
        compute_optical_flow('data/comma_ai/test.mp4', 'data/comma_ai/test_optical_flow')

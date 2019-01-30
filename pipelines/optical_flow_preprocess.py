import os
import cv2
import numpy as np
import pickle
from dask import array as da

def convert_mp4_to_dask_array(input, output, chunksize=500):
    video = cv2.VideoCapture(input)
    chunk_counter = []
    chunk = []
    while video.isOpened():
        valid, frame = video.read()
        if not valid:
            break
        chunk.append(frame)

        if len(chunk) == chunksize:
            with open(os.path.join(output, '{}.npy'.format(len(chunk_counter))), 'w+') as f:
                np.save(f, np.array(chunk))
            print 'Saved Chunk {}'.format(len(chunk_counter))
            chunk_counter.append(len(chunk))
            chunk = []

    with open(os.path.join(output, '{}.npy'.format(len(chunk_counter))), 'w+') as f:
        np.save(f, np.array(chunk))

    video = cv2.VideoCapture(input)
    with open(os.path.join(output, 'info'), 'w+') as f:
        pickle.dump({'chunks': (tuple(chunk_counter), (480,), (640,), (3,)), 'dtype': video.read()[1].dtype, 'axis': 0}, f)
    video.release()

    print 'Split {} into {} chunks'.format(input, chunk_counter)

def compute_optical_flow(input, output):
    input = da.from_npy_stack(input)

    def optical_flow_blocks(block):
        if block.shape == (1,1,1,1):
            return block

        processed_block = []
        prev_frame = cv2.cvtColor(block[0],cv2.COLOR_RGB2GRAY)
        for frame in block:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            processed_block.append(cv2.calcOpticalFlowFarneback(
                prev_frame, frame,
                None, 0.5, 3, 20, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            ))
            print 'Processed: {}'.format(len(processed_block))
        del block
        return np.array(processed_block)

    da.to_npy_stack(
        output,
        input.map_overlap(optical_flow_blocks, (1, 0, 0, 0), boundary='none'),
        axis=0
    )

if __name__ == '__main__':
    convert_mp4_to_dask_array('data/comma_ai/train.mp4', 'data/comma_ai/train')
    convert_mp4_to_dask_array('data/comma_ai/test.mp4', 'data/comma_ai/test')

    import dask
    from multiprocessing.pool import ThreadPool
    with dask.config.set(pool=ThreadPool(1)):
        compute_optical_flow('data/comma_ai/train', 'data/comma_ai/train_optical_flow')
        compute_optical_flow('data/comma_ai/test', 'data/comma_ai/test_optical_flow')

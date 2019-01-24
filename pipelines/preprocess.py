import os
import cv2
import numpy as np
import pickle

def convert_mp4_to_dask_array(input, output, chunksize=1000):
    video = cv2.VideoCapture(input)
    meta = {'chunks': chunksize, 'dtype': None, 'axis': 0}

    chunk_counter = 0
    chunk = []
    while video.isOpened():
        valid, frame = video.read()
        if not valid:
            break
        chunk.append(frame)

        if len(chunk) == chunksize:
            with open(os.path.join(output, 'chunk-{}.npy'.format(chunk_counter)), 'w+') as f:
                np.save(f, np.array(chunk))
            print 'Saved Chunk {}'.format(chunk_counter)
            chunk_counter += 1
            chunk = []

    with open(os.path.join(output, 'chunk-{}.npy'.format(chunk_counter)), 'w+') as f:
        np.save(f, np.array(chunk))

    video = cv2.VideoCapture(input)
    with open(os.path.join(output, 'info'), 'w+') as f:
        pickle.dump({'chunks': chunksize, 'dtype': video.read()[1].dtype, 'axis': 0}, f)
    video.release()

    print 'Split {} into {} chunks'.format(input, chunk_counter)

def unbiased_train_evaluate_split(input, ):
    
if __name__ == '__main__':
    convert_mp4_to_dask_array('data/comma_ai/train.mp4', 'data/comma_ai/train')
    convert_mp4_to_dask_array('data/comma_ai/test.mp4', 'data/comma_ai/test')

import cv2
import dask.array as da
import numpy as np
from keras.models import load_model
from progress.bar import Bar



import os
import cv2
import numpy as np
import pickle
from dask import array as da
from progress.bar import Bar

optical_flow_model = load_model('results/optical_flow_baseline.hdf5')

def compute_speeds(input, output, chunksize=500):
    video = cv2.VideoCapture(input)
    bar = Bar('Processed Frame', max=video.get(cv2.CAP_PROP_FRAME_COUNT))
    speeds = np.array([])
    frames = []

    while video.isOpened():
        valid, frame = video.read()
        if not valid:
            break

        chunk_data.append(frame)
        bar.next()

        if len(chunk_data) == chunksize:
            chunk_data = optical_flow_model.predict(np.array(chunk_data))
            speeds = np.stack([speeds, chunk_data])
            chunk_data = []

    # Save Partial Chunk
    with open(output, 'w+') as f:
        np.save(f, chunk_data)
        chunks.append(len(chunk_data))

    with open(os.path.join(output, 'info'), 'w+') as f:
        pickle.dump({
            'chunks': (tuple(chunks), (chunk_data.shape[1],), (chunk_data.shape[2],), (chunk_data.shape[3],)),
            'dtype': chunk_data.dtype,
            'axis': 0
        }, f)

    video.release()

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    with dask.config.set(pool=ThreadPool(1)):
        compute_segments('data/comma_ai/train.mp4', 'data/comma_ai/train_segments')
        #compute_segments('data/comma_ai/test.mp4', 'data/comma_ai/test_segments')

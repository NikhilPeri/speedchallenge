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

#segmentation_model = load_model('results/upsampling_cnn.hdf5')
def segment_frame(frame):
    frame = segmentation_model.predict(np.array([frame]))[0]

    return segmentation_model.predict(np.array([frame]))[0]

def compute_segments(input, output, chunksize=500):
    video = cv2.VideoCapture(input)
    bar = Bar('Processed Frame', max=video.get(cv2.CAP_PROP_FRAME_COUNT))
    chunks = []
    chunk_data = []

    while video.isOpened():
        valid, frame = video.read()
        if not valid:
            break

        chunk_data.append(frame)
        bar.next()

        if len(chunk_data) == chunksize:
            #chunk_data = segmentation_model.predict(np.array(chunk_data))
            #with open(os.path.join(output, '{}.npy'.format(len(chunks))), 'w+') as f:
                #np.save(f, chunk_data)
            chunks.append(len(chunk_data))
            chunk_data = []

    # Save Partial Chunk
    #with open(os.path.join(output, '{}.npy'.format(len(chunks))), 'w+') as f:
        #chunk_data = segmentation_model.predict(np.array(chunk_data))
        #np.save(f, chunk_data)
    chunks.append(len(chunk_data))

    import pdb; pdb.set_trace()
    with open(os.path.join(output, 'info'), 'w+') as f:
        pickle.dump({
            'chunks': (tuple(chunks), (chunk_data.shape[1],), (chunk_data.shape[2],)),
            'dtype': chunk_data.dtype,
            'axis': 0
        }, f)

    video.release()

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    with dask.config.set(pool=ThreadPool(1)):
        #compute_segments('data/comma_ai/train.mp4', 'data/comma_ai/train_segments')
        compute_segments('data/comma_ai/test.mp4', 'data/comma_ai/test_segments')

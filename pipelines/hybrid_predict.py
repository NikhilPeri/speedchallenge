import cv2
import dask.array as da
import numpy as np
from keras.models import load_model
from dask import array as da

hybrid_model = load_model('results/hybrid_model.hdf5')

def predict(block):
    if block.shape == (1,1,1,1):
        return block
    return hybrid_model.predict(block)

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    with dask.config.set(pool=ThreadPool(8)):
        segments = da.from_npy_stack('data/comma_ai/test_segments')[1:] # drop first segment
        optical_flow = da.from_npy_stack('data/comma_ai/test_optical_flow')

        frame = da.concatenate([optical_flow, segments.reshape((10797, 480, 640, 1))], axis=3)
        predicted_speeds = None
        for block in frame.blocks:
            if predicted_speeds is None:
                predicted_speeds = hybrid_model.predict(block.compute())
            else:
                predicted_speeds = np.concatenate([predicted_speeds, hybrid_model.predict(block.compute())])
            print 'processed block'

        # duplicate first prediction
        with open('test.txt', 'w+') as f:
            np.savetxt(f,predicted_speeds)

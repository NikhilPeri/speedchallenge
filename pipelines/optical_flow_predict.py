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

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    with dask.config.set(pool=ThreadPool(8)):
        optical_flow = da.from_npy_stack('data/comma_ai/test_optical_flow')

        predicted_speeds = None
        for block in optical_flow.blocks:
            if predicted_speeds is None:
                predicted_speeds = optical_flow_model.predict(block.compute())
            else:
                predicted_speeds = np.concatenate([predicted_speeds, optical_flow_model.predict(block.compute())])
            print 'processed block'

        # duplicate first prediction
        with open('test.txt', 'w+') as f:
            np.savetxt(f,predicted_speeds)

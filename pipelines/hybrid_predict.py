import cv2
import dask.array as da
import numpy as np
from keras.models import load_model
from dask import array as da

hybrid_model = load_model('results/optical_flow_baseline.hdf5')

def predict(block):
    if block.shape == (1,1,1,1):
        return block
    return hybrid_model.predict(block)

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    with dask.config.set(pool=ThreadPool(1)):
        #segments = da.from_npy_stack('data/comma_ai/train_segments')
        optical_flow = da.from_npy_stack('data/comma_ai/train_optical_flow')

        #frame = da.concatenate([optical_flow, segments.reshape((20400, 480, 640, 1))], axis=3)
        predicted_speeds = []
        for block in optical_flow.blocks:
            predicted_speeds.append(hybrid_model.predict(block.compute()))
            print 'processed block'
            
        with open('predicted_speed.npy', 'w+') as f:
            np.save(f, np.array(predicted_speeds).flatten())

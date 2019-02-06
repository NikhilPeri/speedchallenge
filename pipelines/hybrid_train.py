import os
import time
import keras
import numpy as np
import pandas as pd
import dask
from multiprocessing.pool import ThreadPool
import dask.array as da

from algorithms.hybrid_model import model
BATCH_SIZE=10
EPOCHS=50

DOWNSAMPLE = 0.2
VALIDATION_SPLIT = 0.3

def fair_split(labels_path, split, downsample):
    speed = np.loadtxt(labels_path, delimiter='\n')
    speed = pd.DataFrame({'speed': speed}).drop(0)
    speed = speed[speed.index % int(1/downsample) == 0]
    speed = speed.sort_values('speed').reset_index().rename(columns={'index': 'frame'})

    training_set = speed[speed.index % int(1/split) != 0]
    validation_set = speed[speed.index % int(1/split) == 0]

    return training_set, validation_set

class DataGenerator(keras.utils.Sequence):
    def __init__(self, labels, optical_flow_path, segments_path, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.optical_flow = da.from_npy_stack(optical_flow_path)
        self.segments = da.from_npy_Stack(segments_path)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    # return a BATCH
    def __getitem__(self, index):
        batch = self.labels.iloc[index*self.batch_size:(index+1)*self.batch_size]
        frames = da.take(self.optical_flow, batch['frame'].values, axis=0).compute()
        segments = da.take(self.segments, batch['frame'].values, axis=0).compute()
        return np.array(frames), batch['speed'].values

    def all(self):
        labels = self.labels.sort_values('frame')
        with dask.config.set(pool=ThreadPool(8)):
            frames = da.take(self.optical_flow, labels['frame'].values, axis=0).compute()
            segments = da.take(self.segments, batch['frame'].values, axis=0).compute()
        import pdb; pdb.set_trace()
        return frames, labels['speed'].values

    def on_epoch_end(self):
        if self.shuffle == True:
            self.labels = self.labels.sample(frac=1.0).reset_index(drop=True)

if __name__ == '__main__':
    timestamp = time.strftime('%c')
    os.mkdir('models/{}'.format(timestamp))
    os.mkdir('logs/{}'.format(timestamp))

    save_callback = keras.callbacks.ModelCheckpoint(
        'models/' + timestamp + '/weights-{val_mean_squared_error:.2f}.hdf5',
        monitor='val_mean_squared_error', verbose=0,
        save_best_only=True, save_weights_only=False,
        mode='auto', period=1
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(timestamp),
        batch_size=BATCH_SIZE,
        update_freq='epoch'
    )

    training_set, validation_set = fair_split('data/comma_ai/train.txt', VALIDATION_SPLIT, DOWNSAMPLE)
    training_data = DataGenerator(training_set, 'data/comma_ai/train_optical_flow').all()
    validation_data = DataGenerator(validation_set, 'data/comma_ai/train_optical_flow').all()
    model.fit(
        *training_data,
        validation_data=validation_data,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[save_callback, tensorboard_callback]
    )

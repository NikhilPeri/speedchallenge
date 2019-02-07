import os
import time
import keras
import numpy as np
import pandas as pd
import dask
from multiprocessing.pool import ThreadPool
import dask.array as da
from sklearn.model_selection import train_test_split

import importlib
from algorithms.hybrid_model import model
BATCH_SIZE=10
EPOCHS=50

speed = np.loadtxt('data/comma_ai/train.txt', delimiter='\n')
speed = pd.DataFrame({'speed': speed}).drop(0)
speed = speed.sort_values('speed').reset_index().rename(columns={'index': 'frame'})

def block_based_split(labels, train_size, test_size, train_block_size, test_block_size, random_state=420):
    labels = labels.sort_values('frame').reset_index(drop=True)

    total_block_size = train_block_size + test_block_size
    block_count = len(labels) / total_block_size

    samples_per_train_block = int(np.floor(train_size * len(labels) / block_count))
    samples_per_test_block = int(np.floor(test_size * len(labels) / block_count))

    train_blocks = []
    test_blocks = []
    for i in range(len(labels) / total_block_size):
        train_block = labels.iloc[i*total_block_size : i*total_block_size + train_block_size]
        test_block = labels.iloc[i*total_block_size + train_block_size : (i + 1)*total_block_size]

        train_blocks.append(train_block.sample(samples_per_train_block, random_state=random_state*i))
        test_blocks.append(test_block.sample(samples_per_test_block, random_state=random_state*i))

    return pd.concat(train_blocks), pd.concat(test_blocks)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, labels, optical_flow_path, segments_path, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.optical_flow = da.from_npy_stack(optical_flow_path)
        self.segments = da.from_npy_stack(segments_path)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    # return a BATCH
    def __getitem__(self, index):
        batch = self.labels.iloc[index*self.batch_size:(index+1)*self.batch_size]
        optical_flow = da.take(self.optical_flow, batch['frame'].values, axis=0).compute()
        segments = da.take(self.segments, batch['frame'].values, axis=0).compute()
        frames = np.concatenate([optical_flow, np.expand_dims(segments, 4)], axis=3)

        return frames, batch['speed'].values

    def all(self):
        labels = self.labels.sort_values('frame')
        with dask.config.set(pool=ThreadPool(8)):
            optical_flow = da.take(self.optical_flow, labels['frame'].values, axis=0).compute()
            segments = da.take(self.segments, labels['frame'].values, axis=0).compute()
        frames = np.concatenate([optical_flow, np.expand_dims(segments, 4)], axis=3)
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

    training_set, validation_set = block_based_split(speed, 0.2, 0.1, 120, 60)
    training_data = DataGenerator(training_set, 'data/comma_ai/train_optical_flow', 'data/comma_ai/train_segments').all()
    validation_data = DataGenerator(validation_set, 'data/comma_ai/train_optical_flow', 'data/comma_ai/train_segments').all()
    while True:
        try:
            model.fit(
                *training_data,
                validation_data=validation_data,
                batch_size=BATCH_SIZE, epochs=EPOCHS,
                callbacks=[save_callback, tensorboard_callback]
            )
        except KeyboardInterrupt:
            keras.backend.clear_session()
            import pdb; pdb.set_trace()
            model = reload(importlib.import_module('algorithms.hybrid_model')).model

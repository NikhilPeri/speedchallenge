import os
import time
import numpy as np
import pandas as pd
import cv2

from algorithms.hybrid_model import model
BATCH_SIZE=500
EPOCHS=10

from progress.bar import Bar

VALIDATION_SPLIT = 0.25

def training_frame(index):
    segmentation_frame = cv2.imread(os.path.join(segmentation, '{}.png'.format(index)))
    optical_flow_frame = optical_flow[index - 1].compute()

    segmentation_frame = np.expand_dims(segmentation_frame[:,:,0]/255., 2)
    return np.concatenate([optical_flow_frame, segmentation_frame], axis=2)

def load_frames(indicies):
    bar = Bar('Loading Frames', max=len(indicies))
    frames = []
    for index in indicies:
        frames.append(training_frame[index])
    return np.array(frames)

if __name__ == '__main__':
    speed = np.loadtxt('data/comma_ai/train.txt', delimiter='\n')
    speed = pd.DataFrame({'speed': speed}).drop(0)
    speed = speed.sort_values('speed').reset_index()

    validation_labels = speed[speed.index % int(1/VALIDATION_SPLIT) == 0]
    validation_frames = load_frames(validation_labels['index'].to_list())
    validation_labels = validation_labels['speed'].to_numpy()

    training_labels = speed[speed.index % int(1/VALIDATION_SPLIT) != 0]
    training_frames = load_frames(training_labels['index'].to_list())
    training_labels = training_labels['speed'].to_numpy()

    timestamp = time.strftime('%c')
    os.mkdir('model/{}'.format(timestamp))
    os.mkdir('logs/{}'.format(timestamp))

    save_callback = keras.callbacks.ModelCheckpoint(
        'model/' + timestamp + '/weights-{val_mean_squared_error:.2f}.hdf5',
        monitor='val_mean_squared_error', verbose=0,
        save_best_only=True, save_weights_only=False,
        mode='auto', period=1
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(timestamp),
        batch_size=BATCH_SIZE,
        update_freq='epoch'
    )
    model.fit(
        training_frames, training_labels,
        validation_data=(validation_frames, validation_labels),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[save_callback, tensorboard_callback]
    )

import os
import cv2
import keras
import numpy as np
from progress.bar import Bar

BATCH_SIZE=500
EPOCHS=10

from algorithms.upsampling_cnn import model

def load_images_and_labels(image_path, label_path):
    image_files = os.listdir(image_path)
    bar = Bar('Loading Images and Labels', max=len(image_files))
    images = []
    labels = []
    for image in image_files:
        images.append(cv2.imread(os.path.join(image_path, image)))
        labels.append(np.load(os.path.join(label_path, image.replace('.jpg', '_train.npy'))))
        bar.next()

    bar.finish()
    return images, labels

if __name__ == '__main__':
    training_images, training_labels = load_images_and_labels('data/bdd100k/stationary/images/train', 'data/bdd100k/stationary/labels/train')
    validation_images, validation_labels = load_images_and_labels('data/bdd100k/stationary/images/val', 'data/bdd100k/stationary/labels/val')
    import pdb; pdb.set_trace()
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
        training_images, training_labels,
        validation_data=(validation_images, validation_labels),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[save_callback, tensorboard_callback]
    )

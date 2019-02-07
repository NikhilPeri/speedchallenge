import time
import os
import cv2
import keras
import numpy as np

MAX_CONCURRENCY=8
BATCH_SIZE=5
EPOCHS=100

from algorithms.upsampling_cnn import model
#model = keras.model.load('models/upsampling.hdf5')

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_path, label_path, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.indexes = np.array([i.replace('.jpg', '') for i in os.listdir(image_path)])
        self.label_path = label_path
        self.image_path = image_path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    # return a BATCH
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        labels = []
        for image in indexes:
            images.append(cv2.imread(os.path.join(self.image_path, image) + '.jpg'))
            labels.append(np.load(os.path.join(self.label_path, image) + '.npy'))

        return np.array(images), np.array(labels)


    def all(self):
        images = []
        labels = []
        for image in self.indexes:
            images.append(cv2.imread(os.path.join(self.image_path, image) + '.jpg'))
            labels.append(np.load(os.path.join(self.label_path, image) + '.npy'))

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

if __name__ == '__main__':

    timestamp = time.strftime('%c')
    os.mkdir('models/{}'.format(timestamp))
    os.mkdir('logs/{}'.format(timestamp))
    save_callback = keras.callbacks.ModelCheckpoint(
        'models/' + timestamp + '/weights-{val_binary_crossentropy:.3f}.hdf5',
        monitor='val_binary_crossentropy', verbose=0,
        save_best_only=True, save_weights_only=False,
        mode='auto', period=1
    )
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(timestamp),
        batch_size=BATCH_SIZE,
        update_freq='epoch'
    )
    model.fit_generator(
        generator=DataGenerator('data/bdd100k/segmentation/processed_images/train', 'data/bdd100k/segmentation/processed_labels/train'),
        validation_data=DataGenerator('data/bdd100k/segmentation/processed_images/val', 'data/bdd100k/segmentation/processed_labels/val').all(),
        use_multiprocessing=True,
        workers=MAX_CONCURRENCY,
        callbacks=[save_callback, tensorboard_callback],
        epochs=EPOCHS
    )

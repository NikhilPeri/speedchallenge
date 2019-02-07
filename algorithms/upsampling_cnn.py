from keras.models import Sequential
from keras.layers import SpatialDropout2D, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape, LocallyConnected2D
from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(480, 640, 3), data_format='channels_last'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2DTranspose(256, (3,3), strides=(2, 2), padding='same', activation='relu'))

model.add(Conv2D(1, (6, 6), padding='same', activation='sigmoid')) # use linearized sigmoid approximation
model.add(UpSampling2D((2,2)))
model.add(Reshape((480, 640)))

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
model.summary()

from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape
from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(480, 640, 3), data_format='channels_last'))
model.add(Dropout(0.25))
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

model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='relu'))

model.add(Conv2D(1, (3, 3), activation='hard_sigmoid', padding='same')) # use linearized sigmoid approximation
model.add(Reshape((240, 320)))

model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='binary_crossentropy', metrics=['mse', 'binary_crossentropy'])
model.summary()

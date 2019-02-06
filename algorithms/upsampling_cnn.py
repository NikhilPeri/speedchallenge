from keras.models import Sequential
from keras.layers import SpatialDropout2D, Dropout, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape, LocallyConnected2D
from keras.optimizers import Adam

model = Sequential()

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(480, 640, 3), data_format='channels_last'))
conv1 = Dropout(0.25))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same'))
conv1 = MaxPooling2D(pool_size=(2, 2)))

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same'))
conv2 = Dropout(0.25))
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same'))
conv2 = MaxPooling2D(pool_size=(2, 2)))

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same'))
conv3 = Dropout(0.25))
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same'))
conv3 = MaxPooling2D(pool_size=(2, 2)))

local1 = MaxPooling2D(pool_size=(2,2))
local1 = LocallyConnected2D(128, 1, activation='relu', use_bias=True)(local1)

model.add()
model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='relu'))

model.add(Conv2D(1, (3, 3), padding='same', activation='relu')) # use linearized sigmoid approximation
model.add(Reshape((480, 640)))

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
model.summary()

from keras.models import Sequential, load_model
from keras import regularizers
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, GaussianNoise, Input, PReLU

from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(4, (4, 4), strides=(4,4), activation='relu', input_shape=(480, 640, 2)))
model.add(Conv2D(4, (4, 4), padding='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Conv2D(8, (4, 4), strides=(4,4), padding='same', activation='relu'))
model.add(Conv2D(8, (4, 4), padding='same', activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(16, (4, 4), strides=(4,4), padding='same', activation='relu'))
model.add(Conv2D(16, (4, 4), padding='same', activation='relu'))
model.add(Dropout(0.1))

model.add(Conv2D(2, (1, 1), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(16, activation='relu', use_bias=True))
model.add(Dense(1, activation='relu'))

model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mse', 'mae'])

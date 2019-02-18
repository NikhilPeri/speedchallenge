from keras.models import Sequential, load_model
from keras import regularizers
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, GaussianNoise, Input, PReLU

from keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(4, (3, 3), strides=(2,2), activation='relu', input_shape=(480, 640, 3)))
model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.6))

model.add(Conv2D(8, (7, 7), strides=(4,4), padding='same', activation='relu'))
model.add(Conv2D(4, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.6))

model.add(Conv2D(16, (7, 7), strides=(4,4), padding='same', activation='relu'))
model.add(Conv2D(8, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(16, (7, 7), strides=(4,4), padding='same', activation='relu'))
model.add(Conv2D(8, (5, 5), padding='same', activation='relu'))

model.add(Conv2D(32, (4, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='relu'))

model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mse', 'mae'])

from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GaussianNoise

from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(480, 640, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(256, (5, 5), activation='relu'))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='mean_squared_error', metrics=['mse'])

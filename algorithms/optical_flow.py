from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Conv2D, ConvLSTM2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten
from keras.optimizers import Adam

BATCH_SIZE=70
EPOCHS=60

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(480,640, 4)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='mean_squared_error', metrics=['mse'])

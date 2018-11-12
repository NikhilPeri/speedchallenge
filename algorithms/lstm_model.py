from keras.models import Sequential
from keras.layers import Dense, Conv2D, ConvLSTM2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from keras.optimizers import Adam

BATCH_SIZE=30
FRAMES_SIZE=20
EPOCHS=1

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (5, 5), padding='same', activation='relu'), input_shape=(FRAMES_SIZE, 480, 640, 1)))
model.add(TimeDistributed(Conv2D(32, (5, 5), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', activation='relu'), input_shape=(FRAMES_SIZE, 480, 640, 1)))
model.add(TimeDistributed(Conv2D(16, (5, 5), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', activation='relu'), input_shape=(FRAMES_SIZE, 480, 640, 1)))
model.add(TimeDistributed(Conv2D(16, (5, 5), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(ConvLSTM2D(32, (5,5), data_format='channels_last', activation='relu'))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(20, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.0, amsgrad=True), loss='mean_squared_error', metrics=['mse'])
model.summary()

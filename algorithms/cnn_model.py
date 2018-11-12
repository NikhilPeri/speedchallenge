from keras.models import Sequential
from keras.layers import Dense, Conv2D, ConvLSTM2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten
from keras.optimizers import Adam

BATCH_SIZE=30
EPOCHS=1

model = Sequential()
model.add(AveragePooling2D(pool_size=(2,2), input_shape=(480, 640, 3)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.0, amsgrad=True), loss='mean_squared_error', metrics=['mse'])

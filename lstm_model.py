from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Conv2D, ConvLSTM2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, TimeDistributed
from keras.optimizers import RMSprop

BATCH_SIZE=200
FRAME_SIZE=2
EPOCHS=100

model = Sequential()
model.add(TimeDistributed(Conv2D(64, (5, 5), padding='same', activation='relu'), input_shape=(FRAME_SIZE, 240, 320, 1)))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(ConvLSTM2D(64, (5,5), data_format='channels_last', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01)))
model.add(Dense(1, activation='linear'))

model.build(input_shape=(None, BATCH_SIZE, FRAME_SIZE, 240, 320, 1))
model.summary()
model.compile(optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0), loss="mean_squared_error", metrics=["mse"])

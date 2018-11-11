from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam

BATCH_SIZE=30
EPOCHS=1

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(480, 640, 3)))
cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
cnn_model.add(Conv2D(32, (5, 5), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(20, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='linear'))

cnn_model.summary()
cnn_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.0, amsgrad=True), loss='mean_squared_error', metrics=['mse'])

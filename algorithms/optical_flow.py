from keras.models import Model
from keras.layers import Input, AveragePooling2D, Conv2D, Concatenate, Dense, Flatten
from keras.optimizers import Adam

current_frame = Input(shape=(480, 640, 3))
previous_frame = Input(shape=(480, 640, 3))
previous_speed = Input(batch_shape=(None, 1))

downsample_current_frame = AveragePooling2D(pool_size=(2,2), strides=(2, 2))(current_frame)
downsample_previous_frame = AveragePooling2D(pool_size=(2,2), strides=(2,2))(previous_frame)

frames = Concatenate(axis=-1)([downsample_current_frame, downsample_previous_frame])
conv_1_frames = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(frames)
conv_2_frames = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(conv_1_frames)
conv_3_frames = Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu')(conv_2_frames)

flat_frames = Flatten()(conv_3_frames)
dense_1 = Dense(4096, activation='relu')(flat_frames)
dense_2 = Dense(4096, activation='relu')(dense_1)
previous_speed_included = Concatenate(axis=-1)([dense_2, previous_speed])
dense_3 = Dense(2048, activation='relu')(previous_speed_included)

output = Dense(1)(previous_speed_included)

model = Model(inputs=[current_frame, previous_frame, previous_speed], outputs=output)
model.summary()
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.0, amsgrad=True), loss='mean_squared_error', metrics=['mse'])

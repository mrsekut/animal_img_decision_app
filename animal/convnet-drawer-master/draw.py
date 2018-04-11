from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense

model = Model(input_shape=(50, 50, 3))

model.add(Conv2D(32, (3 ,3), padding="same"))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(4))

model.save_fig("example.svg")

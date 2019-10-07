import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D


x = np.load('features.npy') # loading x
y = np.load('label.npy') # loading y

x = x / 255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy",
                optimizer = "adam",
                metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=3, validation_split=0.1)

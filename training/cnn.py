# import the necessary packages
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.regularizers import l2


class Model:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        model.add(Convolution2D(filters=32, kernel_size=3, strides=1, input_shape=input_shape,
                                activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu',
                                padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu',
                                padding="same"))
        model.add(MaxPooling2D((2, 2), strides=2))

        model.add(Convolution2D(filters=256, kernel_size=3, strides=1, activation='relu',
                                padding="same", kernel_regularizer=l2(0.01)))
        model.add(MaxPooling2D((2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # output layer

        return model

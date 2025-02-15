'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import cv2
# batch_size = 128
# num_classes = 10
# epochs = 12

# # input image dimensions
# img_rows, img_cols = 28, 28

# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
# y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# #model = load_model('models/my_model.h5')
# #model = Sequential()
# # model.add(Conv2D(32, kernel_size=(3, 3),
# #                  activation='relu',
# #                  input_shape=input_shape))
# # model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
# # model.add(Flatten())
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(num_classes, activation='softmax'))
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
# #               optimizer=tensorflow.keras.optimizers.Adadelta(),
# #               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# model.save('models/my_model2.h5')
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

model = load_model('models/my_model2.h5')
im = cv2.imread("test5.jpg")

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#get thershold image( maybe not used )
test = cv2.resize(im_th[:,:], (28, 28), interpolation=cv2.INTER_AREA)
test = np.reshape(test, (1,28,28,1)).astype(float)

out = model.predict(test)
print(np.argmax(out))



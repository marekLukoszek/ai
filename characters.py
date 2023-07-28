from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
import keras
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(width=28, height=28, max_=None):
    ds, ds_info = tfds.load('emnist/letters', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

    train_ds, test_ds = ds

    def process_data(image, label):
        image = tf.image.resize(image, (width, height))
        image = tf.reduce_mean(image, axis=-1, keepdims=True)  # Zamiana na obrazy w skali szarości
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.one_hot(label, 27)
        return image, label

    train_ds = train_ds.map(process_data)
    test_ds = test_ds.map(process_data)

    if max_ is not None:
        train_ds = train_ds.take(max_)
        test_ds = test_ds.take(max_)

    image_train = np.array([x.numpy() for x, _ in train_ds])
    label_train = np.array([y.numpy() for _, y in train_ds])

    np.save('emnist_train_images.npy', image_train)
    np.save('emnist_train_labels.npy', label_train)

    x_test = np.array([x.numpy() for x, _ in test_ds])
    y_test = np.array([y.numpy() for _, y in test_ds])

    return (image_train, label_train), (x_test, y_test)


def build_net(training_data, width=28, height=28, verbose=False):
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=(height, width, 1),
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(27, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def train(model, training_data, callback=True, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test) = training_data

    if callback == True:
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True,
                                                write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack] if callback else None)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    save_model(model, 'bin/my_emnist_model.h5')


if __name__ == '__main__':
    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(width=28, height=28, max_=None)

    model = build_net(training_data, width=28, height=28, verbose=True)
    train(model, training_data, epochs=10)

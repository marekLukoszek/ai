from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
import keras
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_data_with_custom_image(width=28, height=28, max_=None, verbose=True):
    ds, ds_info = tfds.load('emnist/letters', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

    train_ds, test_ds = ds

    # Load and process the custom image
    custom_image = tf.io.read_file('a_small.jpg')
    custom_image = tf.image.decode_image(custom_image, channels=1)  # Wczytaj obrazek w skali szarości
    custom_image = tf.image.resize(custom_image, (width, height))  # Przeskaluj do wymiarów 28x28 pikseli
    custom_image = tf.cast(custom_image, tf.float32) / 255.0  # Normalizuj piksele (0-1)
    custom_image = tf.expand_dims(custom_image, axis=0)  # Dodaj wymiar wszerz

    custom_label = 0  # Label for the custom image

    # Resize all images from EMNIST to the same shape as the custom image
    def resize_image(image, label):
        image = tf.image.resize(image, (width, height))
        image = tf.expand_dims(image, axis=0)
        return image, label

    train_ds = train_ds.map(resize_image)
    train_ds = train_ds.map(resize_image)
    test_ds = test_ds.map(resize_image)

    # Append the custom image to the training dataset
    train_images, train_labels = [], []
    for image, label in train_ds:
        train_images.append(image)
        train_labels.append(label)
    train_images.append(custom_image)
    train_labels.append(custom_label)

    train_images = tf.concat(train_images)
    train_labels = tf.one_hot(train_labels, 27)  # EMNIST-Letters has 27 classes (26 letters + 1 for unknown)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    if max_ is not None:
        train_ds = train_ds.take(max_)

    test_images = tf.stack([image for image, _ in test_ds])
    test_labels = tf.one_hot([label for _, label in test_ds], 27)

    return (train_ds, train_labels), (test_images, test_labels)


def build_net(training_data, width=28, height=28, verbose=False):
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=(height, width, 27),
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

    if verbose == True:
        print(model.summary())

    return model

def train(model, training_data, callback=True, batch_size=256, epochs=2):
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

    save_model(model, 'bin/model.h5')


if __name__ == '__main__':
    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data_with_custom_image(width=28, height=28, max_=None, verbose=False)

    model = build_net(training_data, width=28, height=28, verbose=False)
    train(model, training_data, epochs=2)

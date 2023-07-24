from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
import keras
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(width=28, height=28, max_=None, verbose=True):
    ds, ds_info = tfds.load('emnist/letters', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

    train_ds, test_ds = ds

    def process_data(image, label):
        image = tf.image.resize(image, (width, height))
        image = tf.reduce_mean(image, axis=-1, keepdims=True)  # Zamiana na obrazy w skali szarości
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.one_hot(label, 27)  # EMNIST-Letters has 27 classes (26 letters + 1 for unknown)
        return image, label

    train_ds = train_ds.map(process_data)
    test_ds = test_ds.map(process_data)

    if max_ is not None:
        train_ds = train_ds.take(max_)
        test_ds = test_ds.take(max_)

    image_train = np.array([x.numpy() for x, _ in train_ds])
    label_train = np.array([y.numpy() for _, y in train_ds])

    x_test = np.array([x.numpy() for x, _ in test_ds])
    y_test = np.array([y.numpy() for _, y in test_ds])

    print(image_train.data.shape)

    unique_classes = np.unique(image_train)
    print(len(unique_classes))
    print(image_train[:0])

    unique_classes = np.unique(label_train)
    print(len(unique_classes))
    print(label_train[:10])

    unique_classes = np.unique(label_train)
    print(len(unique_classes))
    print(unique_classes)

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

    training_data = load_data(width=28, height=28, max_=None, verbose=False)

    model = build_net(training_data, width=28, height=28, verbose=False)
    train(model, training_data, epochs=1)

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
import keras
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from keras.src.saving.saving_api import load_model


def load_data_with_custom_image(width=28, height=28, verbose=True):
    train_images = np.load('emnist_train_images_with_custom_data.npy')
    train_labels = np.load('emnist_train_labels_with_custom_data.npy')

    train_images = train_images.reshape((-1, 28, 28))

    # moj jpg
    custom_image = Image.open('samples/b_B.jpg')
    custom_image = custom_image.resize((28, 28))
    custom_image = custom_image.convert('L')
    custom_image = np.array(custom_image)
    #custom_image = custom_image.astype(np.float32) / 255.0
    negative_image = 1 - custom_image

    custom_label = np.zeros(27)
    custom_label[1] = 1  #indeks to numer kolejności litery w alfabecie

    # print(np.shape(custom_image))
    # print(np.shape(train_images))
    # print(train_labels.shape)
    # print(train_labels[79999])
    # print(custom_label.shape)
    # print(custom_label[0])

    train_images = np.concatenate([train_images, np.expand_dims(negative_image, axis=0)], axis=0)
    train_labels = np.concatenate([train_labels, [custom_label]], axis=0)

    print(np.shape(train_images))
    print(train_labels.shape)

    np.save('emnist_train_images_with_custom_data.npy', train_images)
    np.save('emnist_train_labels_with_custom_data.npy', train_labels)

    return train_images, train_labels


def train(model, train_images, train_labels, callback=True, batch_size=256, epochs=5):
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True,
                                             write_images=True) if callback else None

    model.fit(train_images, train_labels, epochs=epochs,
              verbose=1,
              callbacks=[tbCallBack] if callback else None)

    save_model(model, 'bin/my_emnist_model_with_custom_data.h5')


if __name__ == '__main__':
    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    train_images, train_labels = load_data_with_custom_image(width=28, height=28, verbose=False)

    my_new_model = tf.keras.models.load_model('bin/my_emnist_model_with_custom_data.h5')
    train(my_new_model, train_images, train_labels, True, 5)

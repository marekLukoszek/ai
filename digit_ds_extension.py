import numpy as np
import tensorflow as tf
from PIL import Image
from keras.src.saving.saving_api import save_model

train_images = np.load('mnist_train_images.npy')
train_labels = np.load('mnist_train_labels.npy')

custom_image = Image.open('samples/1.jpg')
custom_image = custom_image.resize((28, 28))
custom_image = custom_image.convert('L')
custom_image = np.array(custom_image)
custom_image = custom_image.astype(np.float32) / 255.0

custom_label = 1
custom_label = np.array(custom_label)

print(custom_image.shape)
print(train_images.shape)
print(train_labels.shape)
print(custom_label.shape)

# Add the custom image and label to the training data
train_images = np.concatenate([train_images, np.expand_dims(custom_image, axis=0)], axis=0)
train_labels = np.concatenate([train_labels, [custom_label]], axis=0)

np.save('mnist_train_images.npy', train_images)
np.save('mnist_train_labels.npy', train_labels)

print(custom_image.shape)
print(train_images.shape)
print(train_labels.shape)
print(custom_label.shape)

train_images = train_images.reshape((-1, 28, 28))

my_new_model = tf.keras.models.load_model('my_mnist_model.h5')

my_new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

my_new_model.fit(train_images, train_labels, epochs=3)

save_model(my_new_model, 'my_mnist_model_with_custom_data.h5')



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import save_model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0  # Corrected test_images scaling

np.save('mnist_train_images.npy', train_images)
np.save('mnist_train_labels.npy', train_labels)

print(train_images.shape)
print(test_images.shape)
print(train_labels[1])

plt.imshow(train_images[1], cmap='gray')
plt.show()

my_model = tf.keras.models.Sequential()
my_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
my_model.add(tf.keras.layers.Dense(128, activation='relu'))
my_model.add(tf.keras.layers.Dense(10, activation='softmax'))

my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Added metrics

my_model.fit(train_images, train_labels, epochs=3)

val_loss, val_acc = my_model.evaluate(test_images, test_labels)
print('Test accuracy:', val_acc)

save_model(my_model, 'my_mnist_model')
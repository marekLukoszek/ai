import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

train_images = np.load('mnist_train_images.npy')
train_labels = np.load('mnist_train_labels.npy')

custom_image = Image.open('samples/1.jpg')
custom_image = custom_image.resize((28, 28))
custom_image = custom_image.convert('L')
custom_image = np.array(custom_image)
custom_image = custom_image.astype(np.float32) / 255.0
custom_image = np.expand_dims(custom_image, axis=0)

my_new_model = tf.keras.models.load_model('my_mnist_model_with_custom_data.h5')

plt.imshow(custom_image[0], cmap='gray')
plt.show()

print(custom_image.shape)
print(train_images.shape)
print(train_labels[60000])

predictions = my_new_model.predict(custom_image)
predicted_label = np.argmax(predictions)

print("Predicted Label:", predicted_label)
import numpy as np

train_images = np.load('emnist_train_images_with_custom_data.npy')
train_labels = np.load('emnist_train_labels_with_custom_data.npy')

print(train_images.shape)
print(train_labels.shape)

print(train_images[60009])
print(train_labels[88800])

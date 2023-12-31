import numpy as np
from PIL import Image
from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt


def predict_letter(model, image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    negative_array = 1 - image_array
    negative_array = np.expand_dims(negative_array, axis=0) / 255.0

    plt.imshow(negative_array[0], cmap='gray')
    plt.show()

    prediction = model.predict(negative_array)
    predicted_label = np.argmax(prediction)
    return predicted_label


if __name__ == '__main__':
    model = load_model('bin/my_emnist_model_with_custom_data.h5')
    predicted_label = predict_letter(model, 'samples/f.jpg')
    mapping = {i: chr(i + 97) for i in range(26)}
    if predicted_label < 26:
        predicted_letter = mapping[predicted_label]
        print('Przewidziana litera:', predicted_letter)
    else:
        print('Nieznana litera, przewidywany indeks:', predicted_label)

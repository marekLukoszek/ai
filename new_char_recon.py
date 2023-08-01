import numpy as np
from PIL import Image
from keras.src.saving.saving_api import load_model
from keras.src.utils import load_img, img_to_array
from matplotlib import pyplot as plt


def predict_letter(model, image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))

    img_array = img_to_array(img)

    img_array /= 255.0

    prepared_img = np.expand_dims(img_array, axis=0)

    print(prepared_img)

    plt.imshow(prepared_img[0], cmap='gray')
    plt.show()

    prediction = model.predict(prepared_img)
    predicted_label = np.argmax(prediction)
    return predicted_label


if __name__ == '__main__':
    model = load_model('bin/my_emnist_model_with_custom_data.h5')
    predicted_label = predict_letter(model, 'samples/a_small.jpg')
    mapping = {i: chr(i + 97) for i in range(26)}
    if predicted_label < 26:
        predicted_letter = mapping[predicted_label]
        print('Przewidziana litera:', predicted_letter)
    else:
        print('Nieznana litera, przewidywany indeks:', predicted_label)

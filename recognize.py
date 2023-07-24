import numpy as np
from PIL import Image
from keras.src.saving.saving_api import load_model


def predict_letter(model, image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 1)
    image_array = image_array.astype('float32') / 255.0
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)
    return predicted_label


if __name__ == '__main__':
    model = load_model('bin/model.h5')
    predicted_label = predict_letter(model, 'a_small.jpg')
    mapping = {i: chr(i + 97) for i in range(26)}  # Mapping for letters (97 is ASCII code for 'a')
    predicted_letter = mapping[predicted_label]
    print('Przewidziana litera:', predicted_letter)
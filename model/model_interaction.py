import keras
import numpy as np

from io import BytesIO
from PIL import Image


model = None
filepath = "model/basesd_model_new.h5"  # "model/weights.h5"
class_names = [
    "Літак",
    "Автомобіль",
    "Птах",
    "Кіт",
    "Олень",
    "Пес",
    "Жаба",
    "Кінь",
    "Корабель",
    "Вантажівка",
]

def load_model():
    """
    The load_model function loads the model from a filepath.

        Args:
            None.

        Returns:
            The loaded model.

    :return: The model
    """
    model = keras.models.load_model(filepath)

    return model


def predict(image: Image.Image):
    """
    The predict function takes an image and returns a string with the predicted class.

    :param image: Image.Image: Pass the image to the predict function
    :return: A string
    """
    global model
    if model is None:
        model = load_model()

    image = np.asarray(image.resize((32, 32)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 255.0

    result = model.predict(image)
    response = class_names[np.array(result).argmax()]

    return f"Class: {response}"


def read_imagefile(file) -> Image.Image:
    """
    The read_imagefile function reads an image file and returns a Image.Image object.

    :param file: Read the image file
    :return: An image object
    """
    image = Image.open(BytesIO(file))

    return image
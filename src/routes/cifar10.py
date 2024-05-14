import httpx as r
from io import BytesIO
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from pathlib import Path

# Path to the directory containing the model
models_dir = Path(__file__).parent.parent.parent.joinpath("models")
# Load the pre-trained model
model = load_model(models_dir.joinpath("cifar10_model_v2.h5"))

# List of labels for the CIFAR-10 dataset
labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def predict(image_url: str):
    """
    Fetch an image from a URL, preprocess it, and make a prediction using the pre-trained model.

    Args:
        image_url (str): URL of the image to be classified.

    Returns:
        numpy.ndarray: Array of prediction probabilities for each class.
    """
    response = r.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB").resize((32, 32))
    img_array = img_to_array(img)
    arr = img_array[None, ...]
    return model.predict(arr)


def fmt_predictions(arr):
    """
    Format the prediction probabilities into a dictionary with labels and their respective probabilities.

    Args:
        arr (numpy.ndarray): Array of prediction probabilities.

    Returns:
        dict: Dictionary with labels as keys and their respective probabilities as values.
    """
    return {
        labels[idx]: round(float(prediction) * 100, 5)
        for idx, prediction in enumerate(arr)
    }
import httpx as r
from io import BytesIO
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from pathlib import Path

models_dir = Path(__file__).parent.parent.parent.joinpath("models")
model = load_model(models_dir.joinpath("cifar10_model_v2.h5"))

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
    response = r.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB").resize((32, 32))
    img_array = img_to_array(img)
    arr = img_array[None, ...]
    return model.predict(arr)


def fmt_predictions(arr):
    return {
        labels[idx]: round(float(prediction) * 100, 5)
        for idx, prediction in enumerate(arr)
    }
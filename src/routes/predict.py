from fastapi import APIRouter
from .cifar10 import predict, fmt_predictions
router = APIRouter()


@router.get("/predict/cifar10")
async def classify_image(url: str):
    """
    Endpoint to classify an image from a given URL using a pre-trained CIFAR-10 model.

    Args:
        url (str): The URL of the image to classify.

    Returns:
        dict: A dictionary with CIFAR-10 labels as keys and their respective probabilities as values.
    """
    predictions = predict(url).flatten()
    return fmt_predictions(predictions)
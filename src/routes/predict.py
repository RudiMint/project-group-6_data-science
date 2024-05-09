from fastapi import APIRouter
from .cifar10 import predict, fmt_predictions
router = APIRouter()


@router.get("/predict/cifar10")
async def classify_image(url: str):
    predictions = predict(url).flatten()
    return fmt_predictions(predictions)
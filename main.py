import io

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pil import Image
from starlette.middleware.cors import CORSMiddleware
import keras
app = FastAPI()

origins = [
    "http://localhost:8000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = keras.models.load_model('basesd_model_new.h5')

# Class mapping
class_names = {
    0: "airplane(літак)",
    1: "automobile(автомобіль)",
    2: "bird(птах)",
    3: "cat(кiт)",
    4: "deer(олень)",
    5: "dog(пес)",
    6: "frog(жаба)",
    7: "horse(кінь)",
    8: "ship(корабель)",
    9: "truck(вантажівка)"
}


@app.get("/")
def read_root():
    return {"message": "Start Page"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    # Check file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    # Read image data
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))  # Adjust the size based on your model's input size
    image_array = np.array(image) / 255.0  # Normalize to [0, 1] range
    if image_array.shape[-1] != 3:  # Ensure RGB
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload an RGB image.")

    # Add a batch dimension and make predictions
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = int(np.argmax(predictions, axis=1)[0])

    return {
        "predicted_class": predicted_class,
        "class_name": class_names[predicted_class]
    }
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

app = FastAPI()

# Load trained model
model = load_model("model/model_info/model/basesd_model_new.h5")

# Define class labels
class_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# # # # 
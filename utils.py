import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def preprocess_image(file, compression_size=64):
    image = Image.open(BytesIO(file)).convert("RGB")
    image = image.resize((compression_size, compression_size))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image
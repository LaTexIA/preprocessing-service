from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import cv2
import numpy as np
import io

app = FastAPI()

@app.post("/preprocess/")
async def preprocess_image(file: UploadFile = File(...)):
    # Lee la imagen que se sube
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convierte la imagen a OpenCV (numpy)
    np_image = np.array(image)
    img_gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(img_gray, (11, 11), 3)

    # Dilatación
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(blurred, kernel, iterations=1)

    # Umbral adaptativo
    img_bin = cv2.adaptiveThreshold(dilated, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 17, 3)

    # Guarda como imagen RGB para usar luego con PIL
    processed_pil = Image.fromarray(processed_image).convert("RGB")
    processed_pil.save(OUTPUT_PATH)


    # Devuelve la imagen procesada
    return FileResponse(output_path, media_type="image/png")

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

    np_image = np.array(image)
    p = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # Aplica un gaussian blurr
    p_blurr  = cv2.GaussianBlur(p, (11, 11), 3)

    # Dilata la imagen para hacer los trazos mas gruesos y uniformes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_dilatada = cv2.dilate(p_blurr, kernel, iterations=1)

    # Convierte la iamgen a blanco y negro (detecta bordes)
    img_bin = cv2.adaptiveThreshold(img_dilatada, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 3)

    #deteccion de bordes
    '''contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    w_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            char_img = img_bin[y:y+h, x:x+w]
            w_images.append((x, char_img))

    w_images.sort(key=lambda x: x[0])'''

    #map(lambda x: cv2.bitwise_not(x), w_images)
    processed_image = cv2.bitwise_not(img_bin) 

    # Guarda la imagen procesada en un archivo temporal
    output_path = "processed_image.png"
    processed_image.save(output_path)

    # Devuelve la imagen procesada
    return FileResponse(output_path, media_type="image/png")

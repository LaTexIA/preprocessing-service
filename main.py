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
    img = await file.read()
    imgsList = re_Process(img)

    return imgsList

def binarizar_y_cuadrar(img, umbral=160, color=255):
    # Aplicar umbral para binarizar la imagen
    img_bin = np.where(img > umbral, 255, 0).astype(np.uint8)

    # Obtener dimensiones
    alto, ancho = img_bin.shape
    tamaño_max = max(ancho, alto)  # Determinar el tamaño cuadrado

    # Crear una imagen cuadrada con el color de padding
    img_cuadrada = np.full((tamaño_max, tamaño_max), color, dtype=np.uint8)

    # Calcular posición para centrar la imagen original
    x_offset = (tamaño_max - ancho) // 2
    y_offset = (tamaño_max - alto) // 2

    # Insertar la imagen binarizada en el centro
    img_cuadrada[y_offset:y_offset+alto, x_offset:x_offset+ancho] = img_bin
    

    return Image.fromarray(img_cuadrada)  # Convertir de vuelta a imagen PIL

def re_Process(img):
    img = Image.open(img)
    img_gray = img.convert("L")
    img_np = np.array(img_gray)

    # Segunda binarización, llegue a 210 como umbral a base de testeo, si se encuentra uno mejor que se use
    img_bin = np.where(img_np > 180, 255, 0).astype(np.uint8)
    
    # Calcular bordes de caracteres
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    w_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        #Solo cuenta caracteres mayores a 1x5px
        if w > 0 and h > 0:
            char_img = img_bin[y:y+h, x:x+w]
            w_images.append((x, char_img))

    # Ordena por coordenada
    w_images.sort(key=lambda x: x[0])

    # return provisional
    return [binarizar_y_cuadrar(img).resize((56,56), Image.Resampling.LANCZOS) for _, img in w_images]

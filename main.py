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
    img_gray = img.convert("L")
    img_np = np.array(img_gray)

    '''# Aplicar umbral para binarizar la imagen
    # Nota, para ciertas imagenes se necesita un umbral mas bajo, valdria la pena hacerlo dinamico
    img_bin = np.where(img_np > 128, 255, 0).astype(np.uint8)

    # Calcular padding
    alto, ancho = img_bin.shape
    tamaño_max = max(ancho, alto)

    # Crear una imagen cuadrada con el color de padding
    img_cuadrada = np.full((tamaño_max, tamaño_max), 255, dtype=np.uint8)

    # Calcular posición para centrar la imagen original
    x_offset = (tamaño_max - ancho) // 2
    y_offset = (tamaño_max - alto) // 2

    img_cuadrada[y_offset:y_offset+alto, x_offset:x_offset+ancho] = img_bin

    # transformar a PIL para aumentar resolucion.
    pil_image = Image.fromarray(img_cuadrada)
    pil_image.resize((1260,1260), Image.Resampling.LANCZOS)

    np_image = np.array(pil_image)

    # Aplicar gaussina blurr para denoising.
    p_blurr  = cv2.GaussianBlur(np_image, (11, 11), 0)

    # Dilatar la imagen para hacer los trazos mas gruesos y uniformes.
    # Esto estoy dudando si dejarlo.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_dilatada = cv2.dilate(p_blurr, kernel, iterations=1)
    '''

    # Segunda binarización, llegue a 210 como umbral a base de testeo, si se encuentra uno mejor que se use
    img_bin = np.where(img_np > 210, 255, 0).astype(np.uint8)
    

    # Calcular bordes de caracteres
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    w_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        #Solo cuenta caracteres mayores a 1x5px
        if w > 1 and h > 5:
            char_img = img_bin[y:y+h, x:x+w]
            w_images.append((x, char_img))

    # Ordena por coordenada
    w_images.sort(key=lambda x: x[0])

    # invierte colores

    # return provisional
    return
    ## HAY QUE ARREGLAR EL TIPO DE RETORNO.
    output_path = "processed_image.bm"
    processed_image.save(output_path)

    # Devuelve la imagen procesada
    return FileResponse(output_path, media_type="image/png")

import base64
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize

app = FastAPI()

@app.post("/preprocess/")
async def preprocess_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    imgsList = re_Process(io.BytesIO(img_bytes))
    return JSONResponse(content={"images": imgsList})

def binarizar_y_cuadrar(img, umbral=160, color=255):
    img_np = np.array(img)
    # Aplicar umbral
    img_bin = np.where(img_np > umbral, 255, 0).astype(np.uint8)

    alto, ancho = img_bin.shape
    tamaño_max = max(ancho, alto)
    img_cuadrada = np.full((tamaño_max, tamaño_max), color, dtype=np.uint8)

    x_offset = (tamaño_max - ancho) // 2
    y_offset = (tamaño_max - alto) // 2

    img_cuadrada[y_offset:y_offset+alto, x_offset:x_offset+ancho] = img_bin

    return Image.fromarray(img_cuadrada)

def re_Process(img):
    pil_img = Image.open(img).convert("L")
    img_np = np.array(pil_img)
    
    # Segunda binarización
    img_bin = np.where(img_np > 180, 255, 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    w_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            # Se extrae la imagen del caracter
            char_img = img_bin[y:y+h, x:x+w]
            w_images.append((x, char_img))
    
    w_images.sort(key=lambda x: x[0])
    
    images_encoded = []
    for _, char_img in w_images:
        # Convierte la imagen recortada a PIL usando binarizar_y_cuadrar y redimensiona a 52x52
        pil_char_img = binarizar_y_cuadrar(Image.fromarray(char_img)).resize((52,52), Image.Resampling.LANCZOS)
        
        # Reducir los trazos usando thinning
        img_array = np.array(pil_char_img)
        # Si es en escala de grises, convertirla a binaria (valores booleanos)
        binary = img_array < 128  # asumiendo fondo blanco
        # Usar skeletonize para obtener el esqueleto
        skeleton = skeletonize(binary)
        # Convertir el resultado a formato imagen (0 o 255)
        thin = (skeleton * 255).astype(np.uint8)
        inverted = 255 - thin
        pil_char_img = Image.fromarray(inverted)
        
        buf = io.BytesIO()
        pil_char_img.save(buf, format="PNG")
        images_encoded.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
    
    if images_encoded:
        images_encoded = images_encoded[1:]
    
    return images_encoded
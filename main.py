from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/preprocess/")
async def preprocess_image(file: UploadFile = File(...)):
    # Lee la imagen que se sube
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Realiza un procesamiento simple: convertir la imagen a escala de grises
    processed_image = image.convert("L")

    # Guarda la imagen procesada en un archivo temporal
    output_path = "processed_image.png"
    processed_image.save(output_path)

    # Devuelve la imagen procesada
    return FileResponse(output_path, media_type="image/png")

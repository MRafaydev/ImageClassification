from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import io
import tensorflow

app = FastAPI()

# Load the model outside the if __name__ block
model = tensorflow.keras.models.load_model("models/breedclassifier.h5")

@app.post("/api/v1/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        img = img.resize((256, 256))  # Resize the image to your target size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        yhat = model.predict(img_array)

        # Process prediction result
        if (yhat > 0.5).any():
            predicted_class = np.argmax(yhat)
            class_labels = ["Gulabi", "Teddy", "MakkahCheeni", "Kamori"]
            predicted_class_name = class_labels[predicted_class]
            return JSONResponse(content={"predicted_class": predicted_class_name}, status_code=200)
        else:
            return JSONResponse(content={"predicted_class": "Wrong"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

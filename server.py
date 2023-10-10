from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import numpy as np

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def main():
    return {"message": "Hello World"}

model =  keras.models.load_model('writing.keras')

@app.post('/predict')
async def predict(data: dict):
    prediction = np.argmax(model.predict(np.array([data["data"]])))
    print(int(prediction))
    return {"prediction" : int(prediction)}
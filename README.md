# Training and Deploying a Neural Network to an API (Using Tensorflow and FastAPI)
  
## Part 1 - Gather Data
##### For the purpose of this guide we will be using the MNIST handwritten digits dataset


The first part of training on any dataset is getting cleaning up the data and getting familiar with it. Luckily for us this dataset is already cleaned and there are no missing values. Here is a summary of the dataset.


Let's start by importing the required modules
```python
#training.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sn
```

Tensorflow's keras already comes with the MNIST dataset split into training and test data so to import it as a numpy array we just need to do the following
```python 
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
```

Once we've separated the test and training data and have the labels for both it's time to normalize the values for the input arrays to values between 0 and 1.

```python
X_train = X_train / 255
X_test = X_test / 255
```


## Part 2 - Create Model

Now that our data is ready it's time to create our model 

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  
```

For this example the model takes a 28 x 28 array as input for the input layer which gets flattened to 784 inputs, has a hidden layer of 100 neurons with a relu activation function and has an output layer of 10 neurons (one for each unique digit in our labels) with a sigmoid activation function. For our optimizer we use the Adam algorithm which you can find more about here https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam. For loss since out labels are just integers one to nine we can use sparse categorical crossentropy and our model will show us accuracy as a metric.

## Part 3 - Train, Evaluate and Save Model

To train the model we can call
```python
model.fit(X_train, y_train, epochs=10)
```
Which trains on our training data for 10 epochs

To test the model and see more detailed results we can use a heatmap of true values and predicted values 

```python
predicted = model.predict(X_test)
predicted_labels = [np.argmax(i) for i in predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=predicted_labels)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```


![heatmap](https://github.com/Avinash64/mnist-recognition-api/blob/master/Pasted%20image%2020230922150048.png?raw=true)

Everything looks good so now it's time to save and export the model
```python
model.save('writing.keras')
```

You should now see a file called writing.keras in the same directory as your python file or notebook.

## Part 4 - Set up the API and import the model

In a new file (I named mine server.py) import the required modules

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import numpy as np
```

Now lets set up fastAPI (for this example I allowed all origins)

```python
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
```

Run the api with uvicorn to make sure everything works

For my file `server.py` the command to run with automatic reloading is 

```bash
uvicorn server:app --reload
```

If you named your file something else just replace server with the name of your file. 

By default the server runs on `http://127.0.0.1:8000` and we can test if it works by opening a command prompt or terminal and running

```bash
curl http://127.0.0.1:8000
```

We should get back the response 

```json
{"message":"Hello World"}
```

Now let's import our model and set up a post route that will accept the input as the value for key "data" in a JSON post body.

```python
model =  keras.models.load_model('writing.keras')

@app.post('/predict')
async def predict(data: dict):
    prediction = np.argmax(model.predict(np.array([data["data"]])))
    return {"prediction" : int(prediction)}
```

Now lets test it out. To make it easier I created an html page with a 28 x 28 canvas that we can draw on and a button that posts the pixel data from the canvas to our route.

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixel Drawing Canvas</title>
    <style>
        canvas {
            border: 1px solid black;
            cursor: crosshair; /* Optional: Change cursor to crosshair for drawing */
        }
    </style>
</head>
<body>
    <canvas id="pixelCanvas" width="280" height="280"></canvas>
    <button onclick="getCanvasData()">Get Canvas Data</button>
    <script>
        const canvas = document.getElementById('pixelCanvas');
        const ctx = canvas.getContext('2d');
  
        let isDrawing = false;
  
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mousemove', drawPixel);
  
        function startDrawing() {
            isDrawing = true;
            drawPixel(event);
        }
  
        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }
  
        function drawPixel(event) {
            if (isDrawing) {
                const rect = canvas.getBoundingClientRect();
                const x = Math.floor((event.clientX - rect.left) / (canvas.width / 28));
                const y = Math.floor((event.clientY - rect.top) / (canvas.height / 28));
  
                ctx.fillStyle = 'black';
                ctx.fillRect(x * (canvas.width / 28), y * (canvas.height / 28), canvas.width / 28, canvas.height / 28);
            }
        }
  
        function getCanvasData() {
            const pixelArray = [];
            for (let y = 0; y < 28; y++) {
                const row = [];
                for (let x = 0; x < 28; x++) {
                    const imageData = ctx.getImageData(x * 10, y * 10, 1, 1).data;
                    const isBlack = imageData[0] === 0 && imageData[1] === 0 && imageData[2] === 0 && imageData[3] !== 0;
                    row.push(isBlack ? 1 : 0);
                }
                pixelArray.push(row);
            }
            postPrediction(pixelArray);
            console.log(pixelArray);
        }
        function postPrediction(data) {
            fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data }),
            })
            .then(response => response.json())
            .then(result => {
                console.log('Post Response:', result);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
```


![[Pasted image 20230922154159.png]]

![[Pasted image 20230922154219.png]]

![[Pasted image 20230922154236.png]]

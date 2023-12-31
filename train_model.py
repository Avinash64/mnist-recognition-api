#train_model.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#load dataset into test and training arrays
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

#normalize data
X_train = X_train / 255
X_test = X_test / 255

#create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  

#train model
model.fit(X_train, y_train, epochs=10)

#evaluate model 
predicted = model.predict(X_test)
predicted_labels = [np.argmax(i) for i in predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=predicted_labels)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#save model
model.save('writing.keras')
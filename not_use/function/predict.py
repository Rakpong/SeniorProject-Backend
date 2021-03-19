import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pathlib
import os

model = tf.keras.models.load_model(
    'F:\workkk\SeniorProject-Backend\Test\model')
# model.summary()

source = "F:/workkk/SeniorProject-Backend/function/frame/"
frame = os.listdir(source)
result = []
# print(frame)
for i in range(14):
    img_path = source + frame[i]
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr/255
    predictions = model.predict(input_arr)
    # print(predictions)

    if(predictions[0][0] > predictions[0][1]):
        result.append("normal")  
    else:
        result.append("smoke")
print(result)

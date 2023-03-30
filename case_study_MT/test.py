import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os
import joblib

def process_output(x):
    x = np.round(x, decimals = 0)
    x = x.astype("uint8")
    return x

def conv_image(np_array):
    return np.moveaxis(np_array, 0, 2)

dir_ = os.getcwd()

#Specify the input image name for the test.!!
#Test image should be in the same directory as this test.py file.

input_image = "raw_RGB_image.tif"

data = rasterio.open(dir_ + '/' + input_image)

model = keras.models.load_model(dir_ + '/' +"model_dense6.keras")

scaler = joblib.load(dir_ + '/' + "scaler.gz")

np_image = data.read()

h = np_image.shape[1]
w = np_image.shape[2]

np_image = np_image.reshape(3,-1).T.astype(int)

np_image = scaler.transform(np_image)

y_pred = model.predict(np_image, batch_size = 4096)

y_pred = process_output((y_pred)).T.reshape(3,h,w)


plt.imshow(conv_image(y_pred))

plt.show()

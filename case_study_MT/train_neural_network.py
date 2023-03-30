import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os

dir_ = os.getcwd()

data = rasterio.open(dir_ + '/' + 'raw_RGB_image.tif')
target = rasterio.open(dir_ + '/' +'true_color_RGB_image.tif')

np_data = data.read()
np_target = target.read()

X = np_data.reshape(3,-1).T.astype(int)
y = np_target.reshape(3,-1).T.astype(int)

scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)

model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(6, activation="relu", kernel_initializer = "normal"))
model.add(layers.Dense(3, kernel_initializer = "normal", use_bias = True))
model.add(layers.ReLU(max_value = 255))

model.compile(optimizer = "rmsprop", loss = "mean_absolute_error")

history = model.fit(X, y, batch_size = 4096, epochs=25, validation_split= 0.3)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.show()


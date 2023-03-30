import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def conv_image(np_array):
    return np.moveaxis(np_array, 0, 2)

def process_output(x):
    x = np.round(x, decimals = 0)
    x[x < 0] = 0
    x[x > 255] = 255
    x = x.astype("uint8")
    return x

def prep_4_predict(x,y):
    X = x.reshape(3,-1).T.astype(int)
    y = y.reshape(3,-1).T.astype(int)
    return X, y

def slice_4_part(np_array):
    c = np_array.shape[0]
    h = np_array.shape[1]
    w = np_array.shape[2]

    up_left = np_array[:, 0:h // 2, 0:w // 2]
    up_right = np_array[:, 0:h // 2, w // 2:]
    down_left = np_array[:, h // 2:, 0:w // 2]
    down_right = np_array[:, h // 2:, w // 2:]

    return up_left, up_right, down_left, down_right

def make_predict(x, y, model):
    y_pred = process_output(model.predict(x))
    error = mean_absolute_error(y, y_pred)
    return y_pred, error

def prep_4_display(x):
    x = x.T
    x = x.reshape(3, -1, 402)
    return x


dir_ = os.getcwd()

data = rasterio.open(dir_ + '/' + 'raw_RGB_image.tif')
target = rasterio.open(dir_ + '/' +'true_color_RGB_image.tif')

np_data = data.read()
np_target = target.read()


train1, train2, train3, train4 = slice_4_part(np_data)
target1, target2, target3, target4 = slice_4_part(np_target)

X1, y1 = prep_4_predict(train1, target1)
X2, y2 = prep_4_predict(train2, target2)
X3, y3 = prep_4_predict(train3, target3)
X4, y4 = prep_4_predict(train4, target4)

#Linear Regression Model
reg = LinearRegression()
reg.fit(X1, y1)

y1_pred, error1 = make_predict(X1, y1, reg)

print("Error 1:", error1)

y2_pred, error2 = make_predict(X2, y2, reg)

print("Error 2:", error2)

y3_pred, error3 = make_predict(X3, y3, reg)

print("Error 3:", error3)

y4_pred, error4 = make_predict(X4, y4, reg)

print("Error 4:", error4)

f, axarr = plt.subplots(1,2)

f.set_figwidth(20)
f.set_figheight(10)

axarr[0].imshow(conv_image(target2))
axarr[0].set_title("True RGB Image")
axarr[1].imshow(conv_image(prep_4_display(y2_pred)))
axarr[1].set_title("Predicted Image ")

plt.show()







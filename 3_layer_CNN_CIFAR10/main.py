import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from model import *
from eval import *

cifar = keras.datasets.cifar10 
(X_train, y_train), (X_test, y_test) = cifar.load_data()

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

## Data augmentation function

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.15),
  tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
  layers.experimental.preprocessing.RandomZoom(height_factor=(0, 0.3), width_factor = (0, 0.3) ),
  layers.experimental.preprocessing.RandomContrast(factor = (0, 2))
])

X_train = np.reshape(X_train, (50000, -1))

##Choosing random 10000 images from training data
np.random.seed(42)
random_indices = np.random.choice(50000, size = 10000, replace = False)
sub_images = X_train[random_indices, :]
sub_images = np.reshape(sub_images, (-1, 32, 32, 3))
sub_labels = y_train[random_indices, :]

## Deleting picked images from original training data
X_train = np.delete(X_train, random_indices, axis = 0)
y_train = np.delete(y_train, random_indices, axis = 0)

## Data augmentation step 
augmented_image = data_augmentation(sub_images)
augmented_image = np.array(augmented_image)

## Reshaping training data
X_train = np.reshape(X_train, (-1, 32, 32, 3))

## Concatenation of augmented images 
X_train = np.concatenate((X_train, augmented_image))
y_train = np.concatenate((y_train, sub_labels))

## Normalizing image data
X_train = X_train / 255
X_test = X_test / 255

## Transforming labels to one-hot vectors
y_train = tf.one_hot(y_train, depth = 10, axis = 1)
y_train = tf.reshape(y_train, shape = (-1,10))
y_test = tf.one_hot(y_test, depth = 10, axis = 1)
y_test = tf.reshape(y_test, shape = (-1,10))

## Specifiying batch size and prepare dataset for tensorflow
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

optimizer = tf.optimizers.Adam( learning_rate=0.0001)

train_accuracy =tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
test_accuracy =tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)

## Defining train and test functions
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model( x, True )
        loss_value = loss1(y, logits)
    grads = tape.gradient(loss_value, weights)
    optimizer.apply_gradients(zip(grads, weights))
    train_accuracy.update_state(y, logits)
    return loss_value
@tf.function
def test_step(x, y):
    test_logits = model(x, False)
    test_loss = loss1(y, test_logits)
    test_accuracy.update_state(y, test_logits)
    return test_loss, test_logits

## Loss and accuracy lists for tracking
epoch_loss_train = []
epoch_acc_train = []
epoch_loss_test = []
epoch_acc_test = []


## Epoch size and training loop
epochs = 0

for epoch in range(epochs):
  batch_loss = []
  batch_loss_test = []
  print("\n Start of epoch %d" % (epoch,))
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
        batch_loss.append( tf.reduce_mean(loss_value).numpy() )
  epoch_train = sum(batch_loss) / len(batch_loss)
  epoch_train = round(epoch_train, 4)  
  epoch_loss_train.append(epoch_train)      
          
  train_accuracy_res = train_accuracy.result()
  epoch_acc_train.append(float(train_accuracy_res))
  print("Training acc over epoch: %.4f" % (float(train_accuracy_res),))
  print("Training loss over epoch: ", epoch_train )
  train_accuracy.reset_states()


  for x_batch_test, y_batch_test in test_dataset:
    temp = test_step(x_batch_test, y_batch_test)
    batch_loss_test.append(tf.reduce_mean(temp[0]).numpy())
  epoch_test = sum(batch_loss_test) / len(batch_loss_test)
  epoch_test = round(epoch_test, 4)  
  epoch_loss_test.append(epoch_test)

  test_accuracy_res = test_accuracy.result()
  epoch_acc_test.append(float(test_accuracy_res))
  print("Validation acc over epoch: %.4f" % (float(test_accuracy_res),))
  print("Validation loss over epoch: ", epoch_test)
  test_accuracy.reset_states()






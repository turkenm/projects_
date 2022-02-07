import tensorflow as tf
import numpy as np
from tensorflow import keras

## Loading trained weights for network
weights = np.load("weights_150epoch_adam_opt.npy", allow_pickle = True)






batch_eval_loss = []
## Testing for loop
for x_batch_test, y_batch_test in test_dataset:
    temp = test_step(x_batch_test, y_batch_test)
    batch_eval_loss.append(tf.reduce_mean(temp[0]).numpy())
epoch_test = sum(batch_eval_loss) / len(batch_eval_loss)
epoch_test = round(epoch_test, 4)

test_accuracy_res = test_accuracy.result()

print("Test accuracy: %.4f" % (float(test_accuracy_res),))
print("Test loss: ", epoch_test)
test_accuracy.reset_states()

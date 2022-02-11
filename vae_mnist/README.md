# README

This project is created for building variational autoencoder model on MNIST dataset. LSTM layer is used for encoder, convolutional layers are used for decoder part of this model. 

## Required Libraries and Versions

* Python Version: 3.7.10
* TensorFlow Version: 2.5.0
* Keras Version: 2.5.0
* Numpy Version: 1.19.5

## main.py

Main.py consist of:
- Loading data set
- Train function
- Training loop
- Loss tracker lists

## network.py
network.py consist of:
- Encoder model
- Decoder model
- Sampling class
- Autoencoder class 
- Loss functions 

Model created subclass of keras.model in order to saving/loading weights and creating generator.


## generator.py
generator.py consist of:
- encoder, decoder and autoencoder model
- Function for plotting generated images

Encoder, decoder and autoencoder models are created for initialize the models then trained weights loaded. (170 epochs)

P.S: All project prepeared on the Google Colab notebooks in order to utilize Google computers that have high computational power. If your computer is not sufficient for deep learning training, it is not recommended to run main.py. You can face with memory problems. __It is recommended to run generator.py only.__ I also put the link of Colab Notebook that all project works properly.

[Google Colab MNIST_VAE_TF_2.X](https://colab.research.google.com/drive/1AfsrFAFIxm3fBT0bB59sXbG2rPrCCotm?usp=sharing)

## Trained Weights
Both .h5 file and native tensorflow extension included to the folder. While loading weights, you can use both "170_epoch_final" and "170_epoch_final.h5"


```python

```

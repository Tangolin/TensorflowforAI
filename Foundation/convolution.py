import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def train_mnist_conv():

    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') >= 0.998:
                print("Reached '99.8%' accuracy so cancelling training!")
                self.model.stop_training = True
    
    Callbacks = myCallback()

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images.reshape(60000,28,28,1)
    test_images = test_images.reshape(10000,28,28,1)
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = keras.models.Sequential([
            keras.layers.Conv2D(64,(3,3),activation = "relu",input_shape=(28,28,1)),
            keras.layers.MaxPool2D(2,2),
            keras.layers.Conv2D(64,(3,3),activation = 'relu'),
            keras.layers.MaxPool2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(128,activation = 'relu'),
            keras.layers.Dense(10,activation = 'softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs = 20, callbacks = [Callbacks])

    model.evaluate(test_images,test_labels)

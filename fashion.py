import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#use a data available in keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()   #load_data returns a tuple

#feature normalising
train_images = train_images / 255.0
test_images = test_images / 255.0

#new model
model = tf.keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28,28)),      #flattens the image into a 1-D array
        keras.layers.Dense(256,activation=tf.nn.relu),  #hidden layer
        keras.layers.Dense(10,activation=tf.nn.softmax) #output layer
    ]
)

model.compile(optimizer= tf.optimizers.Adam(), loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs= 5)

#evaluating the model
model.evaluate(test_images,test_labels)

'''
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
'''

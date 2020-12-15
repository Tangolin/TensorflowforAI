import tensorflow as tf
from tensorflow import keras
import numpy as np

#define class on callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("Reached '99%' accuracy so cancelling training!")
            self.model.stop_training = True

handw = keras.datasets.mnist
(X_train,y_train),(X_test,y_test) = handw.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(256,activation=tf.nn.relu),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

Callbacks = myCallback()

model.fit(X_train,y_train,epochs=10,callbacks=[Callbacks])

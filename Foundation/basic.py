import tensorflow as tf
from tensorflow import keras
import numpy as np

#model creation
model = tf.keras.Sequential(              #creates a backbone for a sequential neural network
    [ 
    keras.layers.Dense(1,input_shape=[1]) #adds a layer with 1 unit and an input of a scalar
    ]                                     #can add more layers within square brackets
)
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')   #how to optimise and the loss function

#providing training data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#fit model(step where the training runs and our weights are changed)
model.fit(xs, ys, epochs=500)   #epochs specify the number of iterations

#prediction
print(model.predict([10]))

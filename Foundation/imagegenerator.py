import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') >= DESIRED_ACCURACY:
                print("Reached '99.9%' accuracy so cancelling training!")
                self.model.stop_training = True
        
    callbacks = myCallback()
    
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150,3)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(16, (3,3), activation = 'relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = RMSprop(lr=0.0009), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    train_datagen = ImageDataGenerator(rescale = 1/255)

    train_generator = train_datagen.flow_from_directory('directory_name', target_size= (150,150), batch_size= 80, class_mode= 'binary')

    model.fit_generator(train_generator, steps_per_epoch= 1, epochs = 25, verbose= 1, callbacks = [callbacks])

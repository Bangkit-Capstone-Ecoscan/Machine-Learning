# Importing library needs
import os
import cv2
import re
import difflib
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


#callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95):
            print("\nAcc dan Val_acc sudah mencapai lebih dari 95%, berhenti training !!!")
            self.model.stop_training = True

# Load the data
data_path = "D:\Dunia Perkuliahan\Semester 5\Bangkit\Food_Image_Recognition\Dataset"
train_path = os.path.join(data_path, "training")
validation_path = os.path.join(data_path, "validation")
evaluation_path = os.path.join(data_path, "evaluation")

categories = sorted(os.listdir(os.path.join(data_path, "training")))


epochs_number = 200

training_datagen = ImageDataGenerator(rescale=1 / 255,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   rotation_range=0.2,
                                   shear_range=0.2,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1 / 255)

training_generator = training_datagen.flow_from_directory(train_path,
                                                    batch_size=64,
                                                    target_size=(150, 150),
                                                    class_mode='categorical',
                                                    shuffle=True
                                                    )

validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                              batch_size=32,
                                                              target_size=(150, 150),
                                                              class_mode='categorical',
                                                              shuffle=False
                                                              )


# Create a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

callback = myCallback()
history = model.fit(training_generator, epochs=epochs_number, validation_data=validation_generator, validation_steps=1, callbacks= callback)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

model.save("model_aug.h5")



"""

Ethan Ali
Sign Language MNIST for Convolutional Networks

Data retrieved from:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

"""

import csv
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

TRAINING_FILE = './sign_mnist_train.csv'
VALIDATION_FILE = './sign_mnist_test.csv'

with open(TRAINING_FILE) as training_file:
    line = training_file.readline()
    print(f"First line (header) looks like this:\n\n{line[:50]} ... {line[-50:]}")
    line = training_file.readline()
    print(f"Each subsequent line (data points) look like this:\n\n{line[:50]} ... {line[-50:]}")

def parse_data_from_input(filename):
  with open(filename) as file:
      
    reader = csv.reader(file, delimiter=',')
    labels = []
    images = []

    line_count = 0
    for row in reader:
        if (line_count != 0):
            labels.append(row[0])
            shaped = np.reshape(row[1:], (28, 28))
            images.append(shaped)
        line_count += 1

    labels = np.array(labels, dtype=float)
    images = np.array(images, dtype=float)
    return images, labels

training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")

def train_val_generators(training_images, training_labels, validation_images, validation_labels):
  
    training_images = np.expand_dims(training_images, axis=3)
    validation_images = np.expand_dims(validation_images, axis=3)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow(x=training_images,
                                       y=training_labels,
                                       batch_size=32)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow(x=validation_images,
                                                 y=validation_labels,
                                                 batch_size=32)

    return train_generator, validation_generator

def create_model():
    # Defining the neural network (Conv2D & MaxPooling2D layers)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(25, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)

model = create_model()

history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator)

# Using matplotlib, plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Created by Ethan Ali

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers,datasets,models
print(tf.__version__)
dataset = datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images /255
test_images = test_images /255

plt.imshow(train_images[15])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (2,2), (1,1), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2,2), (1,1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(760, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#model.fit(train_images, train_labels, batch_size=32, epochs=5, verbose='0')
model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_labels, verbose=1)

pred = model.predict([test_images], verbose=0)

print(np.argmax(pred[1232]))

plt.imshow(test_images[1232])
plt.show()

print(test_labels[1232])

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']#0-9
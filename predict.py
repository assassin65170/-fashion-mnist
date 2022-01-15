import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,datasets
import numpy as np

dataset = datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images /255
test_images = test_images /255

model = models.load_model('model.h5')

model.summary()

pred =model.predict([test_images], verbose=0)

print(np.argmax(pred[111]))

print(test_labels[111])


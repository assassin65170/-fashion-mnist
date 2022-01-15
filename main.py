import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,datasets

dataset = datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images /255
test_images = test_images /255

model = models.load_model('model.h5')

model.summary()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
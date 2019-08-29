import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical

model = keras.models.load_model('pre-trained.h5')
model.summary()

#flatten_layer = model.get_layer(index=7)
#assert flatten_layer.name.startswith('flatten_')

#extractor = keras.models.Model(
    #inputs=model.input, 
    #outputs=flatten_layer.output
#)

# Extract features from private data (unencrypted for now)

x_train_images, y_train = mnist.load_data()[0]
x_train_images = x_train_images.reshape(-1, 28, 28, 1)
x_train_images = x_train_images.astype('float32')
x_train_images /= 255
y_train = to_categorical(y_train, 10)

x_test_images, y_test = mnist.load_data()[1]
x_test_images = x_test_images.reshape(-1, 28, 28, 1)
x_test_images = x_test_images.astype('float32')
x_test_images /= 255
y_test = to_categorical(y_test, 10)

x_train_features = model.predict(x_train_images)
x_test_features  = model.predict(x_test_images)


# Save extracted features for use in fine-tuning
np.save('x_train_features.npy', x_train_features)
np.save('y_train.npy', y_train)

np.save('x_test_features.npy', x_test_features)
np.save('y_test.npy', y_test)
# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.metrics import classification_report
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint


def load_CIFAR_test():
		filename = os.path.join('cifar-10', 'test_batch')
		with open(filename, 'rb') as f:
				datadict = pickle.load(f, encoding='bytes')
		X = datadict[b'data']
		Y = datadict[b'labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
		Y = np.array(Y)
		return X, Y


# Load the datasets
X_test, Y_test = load_CIFAR_test()

# Get a random image for later
random = randint(0,9999)
image = [X_test[random]]

# Use one hot encoding
Y_test = to_categorical(Y_test, 10)

# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='tensorboard/')
model.load("models/model_1/model_1.tfl")

# Get Metrics

target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Y_pred = model.predict(X_test)
# Convert back to labels
Y_test = np.argmax(Y_test, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(Y_test, Y_pred, target_names=target_names))

# Get a random image and show classification
# label = [Y_test[random]]
# predictedLabel = [Y_pred[random]]
# print("Acual: " + target_names[label[0]])
# print("Predicted:" + target_names[predictedLabel[0]])
# plt.imshow(image[0])
# plt.show()

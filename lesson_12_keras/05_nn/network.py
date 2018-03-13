# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

n_classes = np.unique(y_train).size
print('no of classes: {}'.format(n_classes))

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

model = Sequential()

#1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd Layer - Add a fully connected layer
model.add(Dense(128, activation='relu'))

model.add(Dense(n_classes, activation='softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)
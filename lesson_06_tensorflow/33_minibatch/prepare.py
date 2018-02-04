from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from batches_quiz import batches
from pprint import pprint

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('../mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

batches = batches(128, train_features, train_labels)

print("size batch: {b:d}".format(b=len(batches)))

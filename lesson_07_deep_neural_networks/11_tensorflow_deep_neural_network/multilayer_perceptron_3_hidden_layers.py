from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

n_hidden_layer = 256 # layer number of features
n_hidden_layer2 = 1000 # layer number of features
n_hidden_layer3 = 500 # layer number of features

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer, n_hidden_layer2])),
    'hidden_layer3': tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer3])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer3, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer2])),
    'hidden_layer3': tf.Variable(tf.random_normal([n_hidden_layer3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['hidden_layer2'])
layer_2 = tf.nn.relu(layer_2)

# Hidden layer with RELU activation
layer_3 = tf.add(tf.matmul(layer_2, weights['hidden_layer3']), biases['hidden_layer3'])
layer_3 = tf.nn.relu(layer_3)

# Output layer with linear activation
logits = tf.matmul(layer_3, weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Decrease test_size if you don't have enough memory
    test_size = 256
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))

# Epoch: 0001 cost= 28.751916885
# Epoch: 0002 cost= 17.146598816
# Epoch: 0003 cost= 17.940254211
# Epoch: 0004 cost= 6.770376205
# Epoch: 0005 cost= 11.354214668
# Epoch: 0006 cost= 6.313223839
# Epoch: 0007 cost= 27.866151810
# Epoch: 0008 cost= 8.944761276
# Epoch: 0009 cost= 0.000025360
# Epoch: 0010 cost= 0.022511980
# Epoch: 0011 cost= 0.000000064
# Epoch: 0012 cost= 0.000000006
# Epoch: 0013 cost= 5.176272392
# Epoch: 0014 cost= 0.341934204
# Epoch: 0015 cost= 0.000000000
# Epoch: 0016 cost= 9.138504028
# Epoch: 0017 cost= 0.000000000
# Epoch: 0018 cost= 0.000000000
# Epoch: 0019 cost= 5.161387444
# Epoch: 0020 cost= 0.000299055
# Optimization Finished!
# Accuracy: 0.92578125
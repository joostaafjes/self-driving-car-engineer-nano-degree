# Solution is available in the other "solution.py" tab
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

keep_prob = tf.placeholder(tf.float32)

hidden_output = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer_output = tf.nn.relu(hidden_output)
hidden_layer_output = tf.nn.dropout(hidden_layer_output, keep_prob)
output = tf.add(tf.matmul(hidden_layer_output, weights[1]), biases[1])

init = tf.global_variables_initializer()

# Print session results
with tf.Session() as session:
    session.run(init)
    print(session.run(output, feed_dict={keep_prob:0.5} ))


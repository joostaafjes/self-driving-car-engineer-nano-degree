# Solution is available in the other "solution.py" tab
import tensorflow as tf
import numpy as np


def run(factor_data=1):
    output = None
    logit_data = [0.2, 0.1, .01]
    logits = tf.placeholder(tf.float32)
    factor = tf.placeholder(tf.float32)

    # Calculate the softmax of the logits
    softmax = tf.nn.softmax(tf.mul(logits, factor))

    with tf.Session() as sess:
        # Feed in the logit data
        # output = sess.run(softmax, )
        # above is mine -> also working but below is official solution
        output = sess.run(softmax, feed_dict={logits: logit_data, factor: factor_data})

    return output

for factor in np.arange(.1, 10, .1):
    print("factor %1.2f: %s" % (factor, run(factor)))


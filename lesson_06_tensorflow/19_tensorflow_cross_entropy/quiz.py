# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

cross_entropy1 = tf.log(softmax)
cross_entropy2 = tf.mul(one_hot, tf.log(softmax))
cross_entropy3 = tf.sub(0.0, tf.mul(one_hot, tf.log(softmax)))
cross_entropy4 = tf.reduce_sum(tf.sub(0.0, tf.mul(one_hot, tf.log(softmax))))

# Print cross entropy from session
with tf.Session() as session:
    out = session.run(cross_entropy1, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print(out)
    out = session.run(cross_entropy2, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print(out)
    out = session.run(cross_entropy3, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print(out)
    out = session.run(cross_entropy4, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print(out)

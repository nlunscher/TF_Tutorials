from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print "Program Starting..."

import tensorflow as tf
import numpy as np

print "imported"

sess = tf.InteractiveSession()

print "Made Session"

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# to initialize node weights, dont want zeros to start, want some noise
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# initialize convolution layer size
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# initialize pooling layer
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

print "Made helper functions"

# first layer convolution 5x5 patch, 32 features, 1 input channel
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# reshape input x, 28x28 image with 1 colour channel
x_image = tf.reshape(x, [-1,28,28,1]) # cause we falttened it before
# layersetup, convolve layer 1 with input, then ReLu, then maxpool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer, 64 features, 5x5
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer, image is now 7x7, make 1024 outputs
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# use dropout to prevent overfitting when training
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# add an output layer with 10 outputs (each digit)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# do the training
# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# Adam Optimizer instead of gradient decent
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# which predictions where the same as ground truth
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print "Initialize"

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

print "Training"

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
  	# run the test data without dropout (keepprob = 1)
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  # run a minibatch of 50, with keep probability of 0.5
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# test_accs = np.zeros(10)
# for i in range(10):
# 	batch = mnist.test.next_batch(1000)
# 	test_acc = accuracy.eval(feed_dict={
#         x:batch[0], y_: batch[1], keep_prob: 1.0})
# 	test_accs[i] = test_acc
# 	print i, " Test Accuracy: ", test_acc

# overall_test_acc = np.mean(test_accs)
# print "Overall Test Accuracy: ", overall_test_acc

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



save_path = saver.save(sess, "mnist_expert_model.ckplt") # would still need to rebuilt the architecutre, this is just weights



print "Program Ending..."

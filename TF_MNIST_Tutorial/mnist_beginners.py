
# get the data
from tensorflow.examples.tutorials.mnist import input_data
# load the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print "Program Starting..."

# import tensorflow
import tensorflow as tf

# images flattened to 1x784 (from 28x28)
# input data
x = tf.placeholder(tf.float32, [None, 784])

# weights and biases (1 layer fully connected)
# initial values of 0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# output is a softmax ontop of xW + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy (y prime, true output)
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train with gradient decent with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize everything we made
init = tf.initialize_all_variables()

# start the model session and run initialization
sess = tf.Session()
sess.run(init)

# evaluate the model
# how many predictions are the same as ground truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# % of the time the predictions are correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train for 1000 iterations, with batches of 100
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i % 100 == 0:
  	acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
	print i, " Accuracy:", acc



final_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print "Final Accuracy:", final_acc





print "Program Ending..."
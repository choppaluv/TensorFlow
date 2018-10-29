# Lab 7 Learning rate and Evaluation
# Show graph of Accuracy according to learning_rate

import tensorflow as tf
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

lr = tf.placeholder(tf.float32)

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 100

# Accuracy graph according to learning rate
lr_history = []
accuracy_history = []

for i in range(20, 0, -1):
	curr_lr = (i) * 0.1
	lr_history.append(curr_lr)
	with tf.Session() as sess:
		# Initialize TensorFlow variables
		sess.run(tf.global_variables_initializer())
		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0
			total_batch = int(mnist.train.num_examples / batch_size)

			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				c, _ = sess.run([cost, optimizer], feed_dict={
								X: batch_xs, Y: batch_ys, lr: curr_lr})
				avg_cost += c / total_batch


		# Test the model using test sets
		accuracy_val = accuracy.eval(session=sess, feed_dict={
			  X: mnist.test.images, Y: mnist.test.labels, lr: curr_lr})
		print("Learning Rate: ", "{:.2f}".format(curr_lr), "Accuracy: ", accuracy_val)
		accuracy_history.append(accuracy_val)



plt.plot(lr_history, accuracy_history)
plt.show()

# Lab 10 MNIST and NN
# optimal way 
# one hidden layer
# hidden layer size = mean of input and output size
import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 397]))
b1 = tf.Variable(tf.random_normal([397]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([397, 10]))
b2 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L1, W2) + b2

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

'''
Epoch: 0001 cost = 25.788082801
Epoch: 0002 cost = 6.072003106
Epoch: 0003 cost = 3.963279332
Epoch: 0004 cost = 2.864908609
Epoch: 0005 cost = 2.168729554
Epoch: 0006 cost = 1.691331477
Epoch: 0007 cost = 1.307928638
Epoch: 0008 cost = 1.028137531
Epoch: 0009 cost = 0.811458034
Epoch: 0010 cost = 0.653254488
Epoch: 0011 cost = 0.512772904
Epoch: 0012 cost = 0.401356249
Epoch: 0013 cost = 0.326060669
Epoch: 0014 cost = 0.259729231
Epoch: 0015 cost = 0.202165063
Learning Finished!
Accuracy: 0.9474
'''

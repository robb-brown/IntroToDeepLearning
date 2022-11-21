"""
The MIT License (MIT)

Copyright (c) 2015 Robert A. Brown (www.robbtech.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

""" 
This program is very much like perceptronMNIST1 but demonstrates how to clean up
the code and make it ready for training in a loop with a variety of samples by
putting the model creation and cost function calculations in functions.
"""

import tensorflow as tf
import math
from tfs import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define a function that creates our model
def logisticModel(x,W,b):
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	return y

# And a function that calculates the cost
def crossEntropyCost(y,y_):
	crossEntropy = -tf.reduce_sum(y_*tf.math.log(y))
	return crossEntropy

# finally a function that calculates accuracy
def accuracyMetric(y,y_):
	correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
	return accuracy


# Now we can create a function that performs an optimization step
def optimizationStep(optimizer,x,y_,W,b):
	with tf.GradientTape() as tape:
		tape.watch(W); tape.watch(b)
		y = logisticModel(x,W,b)
		cost = crossEntropyCost(y,y_)
	gradient = tape.gradient(target=cost,sources=[W,b])
	optimizer.apply_gradients(zip(gradient, [W, b]))


# And a training function that runs the optimization step repeatedly with a 
# varity of examples
def train(mnist,x,y_,W,b,trainingIterations=3000):
	optimizer = tf.optimizers.Adam(learning_rate=1e-3)
	for iteration in range(trainingIterations):
		# Grab 100 examples instead of just 1. This is called a "minibatch"
		x,y_ = mnist.train.next_batch(100)
		optimizationStep(optimizer=optimizer,x=x,y_=y_,W=W,b=b)
		# For every 100th iteration, print out the training accuracy
		if iteration % 100 == 0:
			y = logisticModel(x,W,b)
			accuracy = accuracyMetric(y,y_)
			print(f'Training accuracy at step {iteration}: {accuracy}')
	

# ---------------  Main Part --------------------
# Now we can use our functions instead of repeating code.
# First, we set up our model variables
W = tf.Variable(tf.zeros([784,10]),name='W_logistic')
b = tf.Variable(tf.zeros([10]),name='b_logistic')

# Check our starting cost and accuracy:
x,y_ = mnist.train.next_batch(1)
y = logisticModel(x,W,b)
crossEntropy = crossEntropyCost(y,y_)
accuracy = accuracyMetric(y,y_)
print(f'Starting Cost: {crossEntropy};  Accuracy: {accuracy}')

# Then we can train
train(mnist=mnist,x=x,y_=y_,W=W,b=b)

# And check our trained cost and accuracy on a batch of held out test examples
x,y_ = mnist.test.next_batch(1)
y = logisticModel(x,W,b)
crossEntropy = crossEntropyCost(y,y_)
accuracy = accuracyMetric(y,y_)
print(f'Trained Cost On Test Set: {crossEntropy};  Accuracy: {accuracy}')


"""
This is better, but our functions are special purpose. If we want to change our
model, especially if we use one with more parameters, we have to rewrite all our
functions. To fix this, we have to create some software infrastructure. Tensorflow
includes a module called Keras which does this, but it also hides a lot of details
inside Tensorflow. We think it's important for learning that you see these details,
so we're going to take the time to create our own minimal system that lets us see
all the parts. See perceptronMNIST3.py.

"""



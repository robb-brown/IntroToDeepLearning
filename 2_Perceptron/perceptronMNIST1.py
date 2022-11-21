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
This program demonstrates how to create a simple model in Tensorflow and perform
an iteration of training.
"""


import tensorflow as tf
import math
from tfs import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Grab an MNIST example. x is the input, y_ is the true label
x,y_ = mnist.train.next_batch(1)


# Set up the variables for our model. This is a basic logistic regression model, where
# W is a tensorflow variable that holds the weights and
# b holds the biases. Note their shapes.
W = tf.Variable(tf.zeros([784,10]),name='W_logistic')
b = tf.Variable(tf.zeros([10]),name='b_logistic')

# For the next two steps we need to tell Tensorflow to keep track of what we're doing
# so it can compute gradients later.
with tf.GradientTape() as tape:
	
	# y is the model's prediction, compare to the true labels, y_
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	
	# Calculate the cost, comparing y to y_
	crossEntropy = -tf.reduce_sum(y_*tf.math.log(y))


# Compute accuracy, for our interest only
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))

# notice that our cost is high and accuracy is low, because the model is not trained.
print(f'Cost: {crossEntropy};  Accuracy: {accuracy}')

# we can train the model by choosing an optimizer,
optimizer = tf.optimizers.Adam(learning_rate=1e-3)

# computing the gradient (that's what that tape thing was for)
# The gradient function computes the partial derivatives of 'target' with
# respect to 'sources'. The resulting vector is the gradient.
# Note that you only get to call this function once!
gradient = tape.gradient(target=crossEntropy,sources=[W,b])

# We can then give our optimizer the gradient and ask it to change our variables
# to move our model's prediction, y closer to the output we desire, y_
# The zip function rearranges our list of gradients and our list of variables into a 
# list of gradient-variable tuples.
optimizer.apply_gradients(zip(gradient, [W, b]))

# We can re-evaluate the cost and accuracy:
y = tf.nn.softmax(tf.matmul(x,W) + b)
newCrossEntropy = -tf.reduce_sum(y_*tf.math.log(y))
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
newAccuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))

# and notice that the cost has decreased a little, and accuracy has increased
print(f'Old Cost: {crossEntropy}; Old Accuracy: {accuracy}')
print(f'New Cost: {newCrossEntropy};  New Accuracy: {newAccuracy}')

# but if we take a different example
x,y_ = mnist.train.next_batch(1)

# and recalculate
y = tf.nn.softmax(tf.matmul(x,W) + b)
newExampleCrossEntropy = -tf.reduce_sum(y_*tf.math.log(y))
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
newExampleAccuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))

# The cost is high for the new example, and accuracy is low again.
print(f'Original Example Cost: {newCrossEntropy},  Accuracy: {newAccuracy}')
print(f'New Example Cost: {newExampleCrossEntropy},  Accuracy: {newExampleAccuracy}')


"""
We'd like the model to do well on all examples, so we need to train it on a
representative sample of them.
Repeatedly re-creating our model every time we want to run a training iteration
or evaluate our result is inelegant and inefficient. Fortunately, we can clean
this up using functions. See perceptronMNIST2.py.
"""

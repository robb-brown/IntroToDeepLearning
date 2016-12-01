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

import tensorflow as tf
import math
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Common setup
sess = tf.InteractiveSession()
x = tf.placeholder('float',shape=[None,784],name='input')		# Input tensor
y_ = tf.placeholder('float', shape=[None,10],name='correctLabels') 		# Correct labels


# MODEL DEFINITION

# Softmax regression
W = tf.Variable(tf.zeros([784,10]),name='W_logistic')
b = tf.Variable(tf.zeros([10]),name='b_logistic')
y = tf.nn.softmax(tf.matmul(x,W) + b)

# # Two-layer softmax regression
# Wlin = tf.Variable(tf.truncated_normal([784,784],stddev= 1.0 / math.sqrt(784)),name='W_linear')
# blin = tf.Variable(tf.zeros([784]),name='b_linear')
# W = tf.Variable(tf.zeros([784,10]),name='W_logistic')
# b = tf.Variable(tf.zeros([10]),name='b_logistic')
# y_intermediate = tf.matmul(x,Wlin) + blin
# y = tf.nn.softmax(tf.matmul(y_intermediate,W) + b)

# # Two-layer perceptron
# hiddenUnitN = 200
# W1 = tf.Variable(tf.truncated_normal([784,hiddenUnitN],stddev= 1.0 / math.sqrt(hiddenUnitN)),name='W1')
# b1 = tf.Variable(tf.zeros([hiddenUnitN]),name='b1')
# y_intermediate = tf.nn.relu(tf.matmul(x,W1) + b1)
# 
# W = tf.Variable(tf.zeros([hiddenUnitN,10]),name='W_logistic')
# b = tf.Variable(tf.zeros([10]),name='b_logistic')
# y = tf.nn.softmax(tf.matmul(y_intermediate,W) + b)

# # Three-layer Deep Network
# hiddenUnitN1 = 200; hiddenUnitN2 = 200
# W1 = tf.Variable(tf.truncated_normal([784,hiddenUnitN1],stddev= 1.0 / math.sqrt(hiddenUnitN1)),name='W1')
# b1 = tf.Variable(tf.zeros([hiddenUnitN1]),name='b1')
# hiddenLayer1 = tf.nn.relu(tf.matmul(x,W1) + b1)
# 
# W2 = tf.Variable(tf.truncated_normal([hiddenUnitN1,hiddenUnitN2],stddev= 1.0 / math.sqrt(hiddenUnitN2)),name='W2')
# b2 = tf.Variable(tf.zeros([hiddenUnitN2]),name='b2')
# hiddenLayer2 = tf.nn.relu(tf.matmul(hiddenLayer1,W2) + b2)
# 
# WSoftmax = tf.Variable(tf.zeros([hiddenUnitN2,10]),name='W_softmax')
# bSoftmax = tf.Variable(tf.zeros([10]),name='b_softmax')
# y = tf.nn.softmax(tf.matmul(hiddenLayer2,WSoftmax) + bSoftmax)


# END MODEL DEFINITION

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

tf.initialize_all_variables().run()		# Take initial values and actually put them in variables

for i in range(1000):						# Do some training
	batch = mnist.train.next_batch(100)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1]})		
		print 'Accuracy at step %d: train: %g' % (i,train_accuracy)
	
	train_step.run(feed_dict={x:batch[0],y_:batch[1]})

print 'Test accuracy: %g' % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})


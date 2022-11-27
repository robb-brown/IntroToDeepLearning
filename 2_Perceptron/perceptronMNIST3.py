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


# This is a base class that defines the basic interface for our objects.
# This one is very basic: an init function that stores the object's
# properties, a setup function that is called when the object is created,
# and an output function that can be called to get the object's output.

class Base(object):
	
	def __init__(self,name):
		self.name = name
		self.setup()
	
	def setup(self):
		pass
	
	def output(self):
		return None
		

# We'll create a 'concrete class' implementing our logistic regression model.
# We have to tell it how big our input and output will be, then it creates
# and stores the W and b variables for us. The output function actually
# computes the logistic regression and returns the result.
# We add a variables function that returns the model's trainable variables because
# our training function will need those.

class LogisticModel(Base):
	
	def __init__(self,inputShape,outputShape,name):
		self.inputShape = inputShape
		self.outputShape = outputShape
		self.name = name
		self.setup()
	
	def setup(self):
		self.W = tf.Variable(tf.zeros([self.inputShape,self.outputShape]),name=self.name+'W_logistic')
		self.b = tf.Variable(tf.zeros([self.outputShape]),name=self.name+'b_logistic')
	
	def variables(self):
		return [self.W,self.b]
	
	def output(self,x):
		return tf.nn.softmax(tf.matmul(x,self.W) + self.b)
	

# Similarly for our cost function. We don't need to create an init function
# because this one will be the same as in the base class.  This object "inherits"
# the init from it's ancestor.
class CrossEntropyCost(Base):

	def output(self,y,y_):
		return -tf.reduce_sum(y_*tf.math.log(y))

# And similarly for our accuracy metric.
class AccuracyMetric(Base):

	def output(self,y,y_):
		correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
		return accuracy

# Our optimizationStep and train function both go into a Trainer object
class Trainer(Base):
	
	def __init__(self,model,cost,metrics,name):
		self.model = model
		self.cost = cost
		self.metrics = metrics
		self.name = name
		self.setup()
		
	def gradient(self,x,y_):
		variables = self.model.variables()
		with tf.GradientTape() as tape:
			y = self.model.output(x=x)
			cost = self.cost.output(y=y,y_=y_)
		gradient = tape.gradient(target=cost,sources=variables)
		return gradient
		
	def train(self,data,iterations=3000,learning_rate=1e-3,batchSize=100):
		optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
		variables = self.model.variables()
		for iteration in range(iterations):
			x,y_ = mnist.train.next_batch(batchSize)
			optimizer.apply_gradients(zip(self.gradient(x,y_),variables))
			if iteration % 100 == 0:
				y = self.model.output(x=x)
				print(f'Training metrics at step {iteration}:')
				print(f'  {self.cost.name} = {self.cost.output(y=y,y_=y_)}')
				for metric in self.metrics:
					print(f'  {metric.name} = {metric.output(y=y,y_=y_)}')
		

# ---------------  Main Part --------------------
# Now we no longer need to create our model by hand in the main part,
# we just create our objects and pass them to the trainer.

model = LogisticModel(inputShape=784,outputShape=10,name='LogisticModel')
cost = CrossEntropyCost(name='CrossEntropy')
metrics = [AccuracyMetric(name='Accuracy')]
trainer = Trainer(model=model,cost=cost,metrics=metrics,name='Trainer')

# Check our starting cost and accuracy:
x,y_ = mnist.train.next_batch(1)
y = model.output(x)
print(f'Starting Cost: {cost.output(y,y_)};  Accuracy: {metrics[0].output(y,y_)}')

# Then we can train
trainer.train(data=mnist)

# And check our trained cost and accuracy on a batch of held out test examples
x,y_ = mnist.test.next_batch(100)
y = model.output(x)
print(f'Trained Cost On Test Set: {cost.output(y=y,y_=y_)};  Accuracy: {metrics[0].output(y=y,y_=y_)}')

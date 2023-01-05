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
This program is very much like perceptronMNIST3 but demonstrates how to implement
different models by first creating the necessary layers objects and then passing those
to a model object.
"""

import tensorflow as tf
import math
from tfs import input_data
import numpy
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# We'll keep our base class from step 3
class Base(object):
	
	def __init__(self,name):
		self.name = name
		self.setup()
	
	def setup(self):
		pass
	
	def output(self):
		return None

# The "deep" part of deep learning refers to the use of sequential computation,
# where the result of one calculation is used as input to another, etc., in a
# a chain. In mathematics, this is function composition: y = h(g(f(x))).
# It's convenient to treat each function as a "layer" of computation, then we
# can connect relatively simple layers together to produce a deep (multi-layered) 
# computational network.

# Our logistic regression model only had one layer of computation. Now we'll introduce
# a layer class, and a model class that will assemble multiple layers. Our 
# old logist regression model will now become a layer, and quite an important one:
# logistic regression is often the final layer in a deep learning classifier.

# This is a 'concrete class' which will be used for creating fairly generic
# fully connected deep learning layer objects. You'll see in the next section
# alternatives to fully connected layers, but for now our models will be
# entirely composed of them.
# A fully connected layer generally has weights (W) and biases (b), and an 
# activation function a, and computes the function:
# y = a(W @ x + b), where x is the input, y is the output and @ is matrix
# multiplication. Our logistic regression layer does exactly this, using 
# the softmax (logistic) function for a. The activation function is important in 
# deep learning because it can be nonlinear. Without it, a model, no matter how big,
# can only learn linear functions, like linear regression does. With a nonlinear
# activation function, a model that is sufficiently large and more than two layers
# deep can learn any function.

# We have to tell out layer how big our input and output will be, and what activation
# function to use. It then creates and stores whatever variables it needs. 
# The output function will assemble the computation for the layer and return it.
# We can also pass in an initialization function, but we'll allow it to default to
# None so that the layer can make a good choice if we don't specify a value. 
# As before, we have a "variables" function that returns the layer's 
# trainable variables, which the model object will collect and pass to 
# the trainer function.
# Since activation functions are very well standardized in Tensorflow, we can actually
# pass the Tensorflow function itself, e.g. tf.identity for a function that does 
# nothing (a linear layer, like linear regression), tf.softmax, which we've seen
# gives us logistic regression, and tf.relu, which is a rectified linear function, the
# workhorse of deep learning.

# For now we'll put all of these classes right in our program so they're easy to
# see, but since we're going to be using them so much, in the next step we'll 
# see how to make our own deep learning module out of them.

class Layer(Base):

	def __init__(self,inputShape,outputShape,activation,name,initialization=None):
		self.inputShape = inputShape
		self.outputShape = outputShape
		self.activation = activation
		self.name = name
		if initialization is None:
			initialization = 'random'
		self.initialization = initialization
		
		self.setup()

	def setup(self):
		if self.initialization == 'random':
			self.W = tf.Variable(tf.random.truncated_normal([self.inputShape,self.outputShape],stddev= 1.0 / math.sqrt(self.outputShape)),name=self.name+'W')
		elif self.initialization == 'zeros':
			self.W = tf.Variable(tf.zeros([self.inputShape,self.outputShape]),name=self.name+'W')
		self.b = tf.Variable(tf.zeros([self.outputShape]),name=self.name+'b')

	def variables(self):
		return [self.W,self.b]

	def output(self,x):
		return self.activation(tf.matmul(x,self.W) + self.b)

# This is our new model class.
# We pass in a list of our layer objects and its job is to connect them for us.
# It also needs to collect the variables from each layer when we ask for them.
# The model doesn't actually need a setup function, but we'll call the ancestor's
# function (which currently does nothing) anyway, because it's good practice.

class Model(Base):

	def __init__(self,layers,name):
		self.layers = layers
		self.name = name
		self.setup()
	
	def variables(self):
		variables = []
		for layer in self.layers:
			variables = variables + layer.variables()
		return variables

	def output(self, x):
		for layer in self.layers:
			x = layer.output(x)
		return x		

# The rest of our functions don't need to change, so we just copy them from
# perceptronMNIST3. In the next step we'll make a module so we don't have to
# do this copying every time!

# ------------------  Unchanged Stuff from before ----------------------
class CrossEntropyCost(Base):

	def output(self,y,y_):
		return -tf.reduce_sum(y_*tf.math.log(y))


class AccuracyMetric(Base):

	def output(self,y,y_):
		correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
		return accuracy


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
			x,y_ = data.train.next_batch(batchSize)
			optimizer.apply_gradients(zip(self.gradient(x,y_),variables))
			if iteration % 100 == 0:
				y = self.model.output(x=x)
				print(f'Training metrics at step {iteration}:')
				print(f'  {self.cost.name} = {self.cost.output(y=y,y_=y_)}')
				for metric in self.metrics:
					print(f'  {metric.name} = {metric.output(y=y,y_=y_)}')

# ---------------------  End of unchanged stuff -------------------------
		

# ---------------  Main Part --------------------
# We can now implement different models by creating layer objects
# and passing them to our model object. We then pass the necessary 
# objects to the trainer.

# Here's our logistic regression model from before:
logistic = Layer(	inputShape=784,outputShape=10,initialization='zeros',
					activation=tf.nn.softmax,name='logistic')
# Notice that even though we only have one layer, we put it in a list.
logisticModel = Model([logistic],name='LogisticModel')


# But now we can just as easily make new models. This one has two layers, a linear one
# then a logistic one. This is like a logistic regression model with interactions
# between variables. We'll just call the layers 1 and 2, but you might see
# terminology calling them "hidden" and "output". The "input" layer is really the
# input to layer 1. The output of layer 1 is "hidden" because it's not data
# we observe, or the end result of the computation, it's an intermediate result.
# Of course, we can see it just fine, and we'll look at it later.

# Two-layer logistic Regression Model:
layer1 = Layer(	inputShape=784,outputShape=200,initialization='random',
				activation=tf.identity,name='linear')

layer2 = Layer(	inputShape=200,outputShape=10,initialization='zeros',
				activation=tf.nn.softmax,name='logistic')

twoLayerRegressionModel = Model([layer1,layer2],name='TwoLayerRegressionModel')


# This one is called a multilayer perceptron (MLP). It has two layers, just like the
# previous one, but notice that the first layer has a proper activation function;
# it's nonlinear! Our model is very small, but, if scaled up by increasing
# the size of outputShape on the first layer and inputShape on the second, 
# this model has everything it needs to learn *any* function.

# Two-layer Perceptron Model:
layer1 = Layer(	inputShape=784,outputShape=200,initialization='random',
				activation=tf.nn.relu,name='input')

layer2 = Layer(	inputShape=200,outputShape=10,initialization='zeros',
				activation=tf.nn.softmax,name='output')

twoLayerPerceptronModel = Model([layer1,layer2],name='TwoLayerPerceptron')


# Finally, this is pretty much the same thing, but it's got an extra layer in the
# middle, for three total! We're officially doing deep learning now! Most machine
# learning models, like decision trees, SVM, regression, 
# older neural networks, etc. are one or two layers of computation deep.

# Three-Layer Deep Network Model:
layer1 = Layer(	inputShape=784,outputShape=100,initialization='random',
				activation=tf.nn.relu,name='input')

layer2 = Layer(	inputShape=100,outputShape=100,initialization='random',
				activation=tf.nn.relu,name='hidden')

layer3 = Layer(	inputShape=100,outputShape=10,initialization='zeros',
				activation=tf.nn.softmax,name='output')

threeLayerPerceptronModel = Model([layer1,layer2,layer3],name='ThreeLayerDeepModel')


# Now we can set up our cost function and trainer as we did before.
cost = CrossEntropyCost(name='CrossEntropy')
metrics = [AccuracyMetric(name='Accuracy')]
trainer = Trainer(model=logisticModel,cost=cost,metrics=metrics,name='Trainer')


# We can train our model just as we did before 
# (remember, we didn't change the training function at all)
# but let's do a little experiment. Since we now have multiple models,
# let's train them all and see which one works best!

# We'll create a dictionary to hold our results, then loop over 
# our models to train each one in succession.

results = dict()
for model in [logisticModel,twoLayerRegressionModel,twoLayerPerceptronModel,threeLayerPerceptronModel]:
	print(f'Training model {model.name}')

	# We'll create another dictionary to store the result just for this model
	result = dict()
	
	# we need to (re)create our trainer inside the loop with the current model
	# we don't need to recreate the cost or metrics, we can reuse those.
	trainer = Trainer(model=model,cost=cost,metrics=metrics,name='Trainer')
	
	# Check our starting cost and accuracy:
	x,y_ = mnist.train.next_batch(100)
	y = model.output(x)
	result['startingCost'] = cost.output(y,y_)
	result['startingAccuracy'] = metrics[0].output(y,y_)
	print(f'Starting Cost: {result["startingCost"]};  Accuracy: {result["startingAccuracy"]}')	
	
	# Then we can train
	trainer.train(data=mnist)

	# And check our trained cost and accuracy on a batch of held out test examples
	x,y_ = mnist.test.next_batch(100)
	y = model.output(x)
	result['testCost'] = cost.output(y,y_)
	result['testAccuracy'] = metrics[0].output(y,y_)
	print(f'Trained Cost On Test Set: {result["testCost"]};  Accuracy: {result["testAccuracy"]}')

	# finally, we'll put our result dictionary into our results dictionary so
	# we can analyze it later.
	results[model.name] = result


# Now that we have all our results, let's print them out so they're easier to see
# all together. If you look closely, I've added :0.4 after each variable we're
# printing. This tells python to use four digits of precision. It makes the printout
# a little easier to read.

for modelName in results.keys():
	result = results[modelName]
	print()
	print(modelName)
	print(f'Starting Cost: {result["startingCost"]:0.4};  Accuracy: {result["startingAccuracy"]:0.4}')	
	print(f'Trained Cost On Test Set: {result["testCost"]:0.4};  Accuracy: {result["testAccuracy"]:0.4}')


# Congratulations, you have just completed a deep learning experiment. So, which
# model performed the best?  Here's what I got:

# LogisticModel
# Starting Cost: 230.3;  Accuracy: 0.12
# Trained Cost On Test Set: 20.74;  Accuracy: 0.97
#
# TwoLayerRegressionModel
# Starting Cost: 230.3;  Accuracy: 0.1
# Trained Cost On Test Set: 15.77;  Accuracy: 0.96
#
# TwoLayerPerceptron
# Starting Cost: 230.3;  Accuracy: 0.13
# Trained Cost On Test Set: 9.695;  Accuracy: 0.98
#
# ThreeLayerDeepModel
# Starting Cost: 230.3;  Accuracy: 0.11
# Trained Cost On Test Set: 6.637;  Accuracy: 0.97

# Notice that all the models did pretty well on accuracy, but remember, we only
# tested with a batch of 100 examples. The cost value takes into account
# how confident each of the predictions was, and shows that the two layer models
# were better than one layer, and the three layer model was better than two, and
# the nonlinearity in the perceptrons also helped. If you were paying attention you
# might have noticed that, although the one-layer logistic model's size is fixed
# by the input and output, all the two layer models had 200 hidden artificial neurons
# and the three layer model also had 200 hidden neurons, 100 in each hidden layer. 
# This actually means the three layer model had fewer parameters, because W is a
# function of inputSize * outputSize. But the three layer model did the best!
# There's a proof that shows that a deeper model can use it's capacity more
# efficiently (sometimes exponentially more efficiently) than a shallower model.
# That's why deep learning works so well.

# Three layers doesn't seem like much, but without ReLU and functions like it, three or
# more layer models are very hard to train. ReLU is really the magic that enables deep
# learning, and the efficiency of deep models is the reason deep learning works so well.

# **An important note!**
# You might have noticed that the starting costs can be a bit different, and the
# starting accuracies can be quite different. If you run this program again, you might
# get somewhat different results, and you probably got different results than I did.
# There are a lot of random numbers in machine learning. We used random numbers to
# intialize many of our parameters, and also chose random examples, in a random
# order, to train and test our models. To be good scientists, we should run our program
# many times, keep track of the results for each run, and compute statistics that
# tell us which models were significantly better or worse. Never believe a number
# or that doesn't have some kind of confidence interval! You may also notice that
# lots of machine learning websites, competitions and even scientific papers
# don't do this. This is really an area where the machine learning field needs
# to improve. If you'd like to help, try modifying this program to train the
# models multiple times, collect the results, and do a statistical comparison. Don't
# forget to recreate your models each time so they're untrained! Do our initial results
# still hold up?





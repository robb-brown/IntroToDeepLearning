import tensorflow as tf
import math
from tfs import input_data
import numpy
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# We've been importing input_data from the tfs module in order to provide
# the mnist data. This module also contains the classes we've been writing
# to support our model building, so instead of including this support code
# in every program, we can just import it:

from tfs import Layer, Model, CrossEntropyCost, AccuracyMetric, Trainer


# Throughout this tutorial we'll introduce new tools in the main program
# on first use so you can always see all the details, then subsequently 
# import them from tfs.


# ---------------  Main Part --------------------

# Logistic regression model
logistic = Layer(	inputShape=784,outputShape=10,initialization='zeros',
					activation=tf.nn.softmax,name='logistic')
logisticModel = Model([logistic],name='LogisticModel')


# Two-layer logistic Regression Model:
layer1 = Layer(	inputShape=784,outputShape=200,initialization='random',
				activation=tf.identity,name='linear')
layer2 = Layer(	inputShape=200,outputShape=10,initialization='zeros',
				activation=tf.nn.softmax,name='logistic')
twoLayerRegressionModel = Model([layer1,layer2],name='TwoLayerRegressionModel')


# Two-layer Perceptron Model:
layer1 = Layer(	inputShape=784,outputShape=200,initialization='random',
				activation=tf.nn.relu,name='input')
layer2 = Layer(	inputShape=200,outputShape=10,initialization='zeros',
				activation=tf.nn.softmax,name='output')
twoLayerPerceptronModel = Model([layer1,layer2],name='TwoLayerPerceptron')


# Three-Layer Deep Network Model:
layer1 = Layer(	inputShape=784,outputShape=100,initialization='random',
				activation=tf.nn.relu,name='input')
layer2 = Layer(	inputShape=100,outputShape=100,initialization='random',
				activation=tf.nn.relu,name='hidden')
layer3 = Layer(	inputShape=100,outputShape=10,initialization='zeros',
				activation=tf.nn.softmax,name='output')
threeLayerPerceptronModel = Model([layer1,layer2,layer3],name='ThreeLayerDeepModel')


# Cost and trainer
cost = CrossEntropyCost(name='CrossEntropy')
metrics = [AccuracyMetric(name='Accuracy')]
trainer = Trainer(model=logisticModel,cost=cost,metrics=metrics,name='Trainer')



# Train and evaluate
results = dict()
for model in [logisticModel,twoLayerRegressionModel,twoLayerPerceptronModel,threeLayerPerceptronModel]:
	print(f'Training model {model.name}')
	result = dict()
	trainer = Trainer(model=model,cost=cost,metrics=metrics,name='Trainer')
	x,y_ = mnist.train.next_batch(100)
	y = model.output(x)
	result['startingCost'] = cost.output(y,y_)
	result['startingAccuracy'] = metrics[0].output(y,y_)
	print(f'Starting Cost: {result["startingCost"]};  Accuracy: {result["startingAccuracy"]}')	
	trainer.train(data=mnist)
	x,y_ = mnist.test.next_batch(100)
	y = model.output(x)
	result['testCost'] = cost.output(y,y_)
	result['testAccuracy'] = metrics[0].output(y,y_)
	print(f'Trained Cost On Test Set: {result["testCost"]};  Accuracy: {result["testAccuracy"]}')
	results[model.name] = result


for modelName in results.keys():
	result = results[modelName]
	print()
	print(modelName)
	print(f'Starting Cost: {result["startingCost"]:0.4};  Accuracy: {result["startingAccuracy"]:0.4}')	
	print(f'Trained Cost On Test Set: {result["testCost"]:0.4};  Accuracy: {result["testAccuracy"]:0.4}')

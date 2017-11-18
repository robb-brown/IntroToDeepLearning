"""The MIT License (MIT)

Copyright (c) 2016 Robert A. Brown (www.robbtech.com)

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
import pylab as mpl
import numpy as np
import time


def weightVariable(shape,std=1.0,name=None):
	# Create a set of weights initialized with truncated normal random values
	name = 'weights' if name is None else name
	return tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=std/math.sqrt(shape[0])))

def biasVariable(shape,bias=0.1,name=None):
	# create a set of bias nodes initialized with a constant 0.1
	name = 'biases' if name is None else name
	return tf.get_variable(name,shape,initializer=tf.constant_initializer(bias))

def conv2d(x,W,name=None):
	# return an op that convolves x with W
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name=name)

def conv3d(x,W,name=None):
	# return an op that convolves x with W
	return tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME',name=name)

def max_pool_2x2(x,name=None):
	# return an op that performs max pooling across a 2D image
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

def max_pool(x,shape,name=None):
	# return an op that performs max pooling across a 2D image
	return tf.nn.max_pool(x,ksize=[1]+shape+[1],strides=[1]+shape+[1],padding='SAME',name=name)

def max_pool3d(x,shape,name=None):
	# return an op that performs max pooling across a 2D image
	return tf.nn.max_pool3d(x,ksize=[1]+shape+[1],strides=[1]+shape+[1],padding='SAME',name=name)
	
	
def plotFields(layer,fieldShape=None,channel=None,figOffset=1,cmap=None,padding=0.01):
	# Receptive Fields Summary
	W = layer.W
	wp = W.eval().transpose();
	if len(np.shape(wp)) < 4:		# Fully connected layer, has no shape
		fields = np.reshape(wp,list(wp.shape[0:-1])+fieldShape)	
	else:			# Convolutional layer already has shape
		features, channels, iy, ix = np.shape(wp)
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = np.reshape(wp,[features*channels,iy,ix])

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))

	fig = mpl.figure(figOffset); mpl.clf()
	
	# Using image grid
	from mpl_toolkits.axes_grid1 import ImageGrid
	grid = ImageGrid(fig,111,nrows_ncols=(perRow,perColumn),axes_pad=padding,cbar_mode='single')
	for i in range(0,np.shape(fields)[0]):
		im = grid[i].imshow(fields[i],cmap=cmap); 

	grid.cbar_axes[0].colorbar(im)
	mpl.title('%s Receptive Fields' % layer.name)
	
	# old way
	# fields2 = np.vstack([fields,np.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])
	# tiled = []
	# for i in range(0,perColumn*perRow,perColumn):
	# 	tiled.append(np.hstack(fields2[i:i+perColumn]))
	# 
	# tiled = np.vstack(tiled)
	# mpl.figure(figOffset); mpl.clf(); mpl.imshow(tiled,cmap=cmap); mpl.title('%s Receptive Fields' % layer.name); mpl.colorbar();
	mpl.figure(figOffset+1); mpl.clf(); mpl.imshow(np.sum(np.abs(fields),0),cmap=cmap); mpl.title('%s Total Absolute Input Dependency' % layer.name); mpl.colorbar()



def plotOutput(layer,feed_dict,fieldShape=None,channel=None,figOffset=1,cmap=None):
	# Output summary
	W = layer.output
	wp = W.eval(feed_dict=feed_dict);
	if len(np.shape(wp)) < 4:		# Fully connected layer, has no shape
		temp = np.zeros(np.product(fieldShape)); temp[0:np.shape(wp.ravel())[0]] = wp.ravel()
		fields = np.reshape(temp,[1]+fieldShape)
	else:			# Convolutional layer already has shape
		wp = np.rollaxis(wp,3,0)
		features, channels, iy,ix = np.shape(wp)
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = np.reshape(wp,[features*channels,iy,ix])

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))
	fields2 = np.vstack([fields,np.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])
	tiled = []
	for i in range(0,perColumn*perRow,perColumn):
		tiled.append(np.hstack(fields2[i:i+perColumn]))

	tiled = np.vstack(tiled)
	if figOffset is not None:
		mpl.figure(figOffset); mpl.clf(); 

	mpl.imshow(tiled,cmap=cmap); mpl.title('%s Output' % layer.name); mpl.colorbar();



def train(session,trainingData,testingData,input,truth,cost,trainingStep,accuracy,iterations=5000,miniBatch=100,trainDict={},testDict=None,logName=None,initialize=True,addSummaryOps=True):
	testDict = trainDict if testDict is None else testDict
	
	if addSummaryOps:
		costSummary = tf.summary.scalar("Cost Function", cost)
		if accuracy is None:
			accuracy = cost
		accuracySummary = tf.summary.scalar("accuracy", accuracy)
		mergedSummary = tf.summary.merge_all()
		if logName is not None:
			writer = tf.train.SummaryWriter(logName, session.graph_def)

	if initialize:
		tf.global_variables_initializer().run()		# Take initial values and actually put them in variables

	lastTime = 0; lastIterations = 0
	print("Doing {} iterations".format(iterations))
	for i in range(iterations):						# Do some training
		batch = trainingData.next_batch(miniBatch)
		if (i%100 == 0) or (time.time()-lastTime > 5):
			
			testDict.update({input:batch[0],truth:batch[1]})
#			trainAccuracy = accuracy.eval(feed_dict=testDict)

			# Test accuracy for TensorBoard
#			testDict.update({input:testingData.images,truth:testingData.labels})
			if addSummaryOps:
				summary,testAccuracy = session.run([mergedSummary,accuracy],feed_dict=testDict)
				if logName is not None:
					writer.add_summary(summary,i)
			else:
				testAccuracy = session.run([accuracy],feed_dict=testDict)[0]

			print('Accuracy at batch {}: {} ({} samples/s)'.format(i,testAccuracy,(i-lastIterations)/(time.time()-lastTime)*miniBatch))
			lastTime = time.time(); lastIterations = i

		trainDict.update({input:batch[0],truth:batch[1]})
		trainingStep.run(feed_dict=trainDict)

	try:
		# Only works with mnist-type data object
		testDict.update({input:testingData.images, truth:testingData.labels})
		print('Test accuracy: {}'.format(accuracy.eval(feed_dict=testDict)))
	except:
		pass
	




class Layer(object):
	
	def __init__(self,input,units,name,std=1.0,bias=0.1):
		self.input = input
		self.units = units
		self.name = name
		self.initialize(std=std,bias=bias)
		self.setupOutput()
		self.setupSummary()
		
	def initialize(self):
		pass
		
	def setupOutput(self):
		pass
		
	def setupSummary(self):
		pass
		
		
class UtilityLayer(Layer):

	def __init__(self,input,name):
		self.input = input
		self.name = name
		self.initialize()
		self.setupOutput()
		self.setupSummary()


class Linear(Layer):
	
	def initialize(self,std=1.0,bias=0.1):
		with tf.variable_scope(self.name):
			self.inputShape = np.product([i.value for i in self.input.get_shape()[1:] if i.value is not None])
			self.W = weightVariable([self.inputShape,self.units],std=std)
			self.b = biasVariable([self.units],bias=bias)

	def setupOutput(self):
		if len(self.input.get_shape()) > 2:
			input = tf.reshape(self.input,[-1,self.inputShape])	# flatten reduced image into a vector
		else:
			input = self.input
		self.output = tf.matmul(input,self.W)

	def setupSummary(self):
		self.WHist = tf.histogram("weights" % self.name, self.W)
		self.BHist = tf.histogram("biases" % self.name, self.b)
		self.outputHist = tf.histogram("output" % self.name, self.output)


		
class SoftMax(Layer):
	
	def initialize(self,std=1.0,bias=0.1):
		with tf.variable_scope(self.name):
			self.inputShape = np.product([i.value for i in self.input.get_shape()[1:] if i.value is not None])
			self.W = weightVariable([self.inputShape,self.units],std=std)
			self.b = biasVariable([self.units],bias=bias)
				
	def setupOutput(self):
		if len(self.input.get_shape()) > 2:
			input = tf.reshape(self.input,[-1,self.inputShape])	# flatten reduced image into a vector
		else:
			input = self.input
		self.output = tf.nn.softmax(tf.matmul(input,self.W) + self.b)

	def setupSummary(self):
		self.WHist = tf.summary.histogram("weights", self.W)
		self.BHist = tf.summary.histogram("biases", self.b)
		self.outputHist = tf.summary.histogram("output", self.output)


class ReLu(SoftMax):

	def setupOutput(self):
		if len(self.input.get_shape()) > 2:
			input = tf.reshape(self.input,[-1,self.inputShape])	# flatten reduced image into a vector
		else:
			input = self.input
		self.output = tf.nn.relu(tf.matmul(input,self.W) + self.b)


class Conv2D(SoftMax):

	def __init__(self,input,shape,name,std=1.0,bias=0.1):
		self.input = input
		self.units = shape[-1]
		self.shape = shape
		self.name = name
		self.initialize(std=std,bias=bias)
		self.setupOutput()
		self.setupSummary()

	
	def initialize(self,std=1.0,bias=0.1):
		with tf.variable_scope(self.name):
			self.W = weightVariable(self.shape,std=std)		# YxX patch, Z contrast, outputs to N neurons
			self.b = biasVariable([self.shape[-1]],bias=bias)	# N bias variables to go with the N neurons

	def setupOutput(self):
		self.output = tf.nn.relu(conv2d(self.input,self.W) + self.b)


class ConvSoftMax(Conv2D):

	def setupOutput(self):
		inputShape = self.input.get_shape()
		convResult = conv2d(self.input,self.W) + self.b

		convResult = tf.reshape(convResult,[-1,self.units])	# flatten reduced image into a vector
		softMaxed = tf.nn.softmax(convResult)
		self.output = tf.reshape(softMaxed,[-1] + inputShape[1:3].as_list() + [self.units])


class Conv3D(Conv2D):
	
	def setupOutput(self):
		self.output = tf.nn.relu(conv3d(self.input,self.W) + self.b)


class Conv3DSoftMax(ConvSoftMax):

	def setupOutput(self):
		inputShape = self.input.get_shape()
		convResult = conv3d(self.input,self.W) + self.b

		convResult = tf.reshape(convResult,[-1,self.units])	# flatten reduced image into a vector
		softMaxed = tf.nn.softmax(convResult)
		self.output = tf.reshape(softMaxed,[-1] + inputShape[1:4].as_list() + [self.units])



class MaxPool2x2(UtilityLayer):

	def setupOutput(self):
		with tf.variable_scope(self.name):
			self.output = max_pool_2x2(self.input)
			


class MaxPool(UtilityLayer):
	
	def __init__(self,input,shape,name):
		self.shape = shape
		super(MaxPool,self).__init__(input,name)

	def setupOutput(self):
		with tf.variable_scope(self.name):
			self.output = max_pool(self.input,shape=self.shape)
			

class MaxPool3D(MaxPool):

	def setupOutput(self):
		with tf.variable_scope(self.name):
			self.output = max_pool3d(self.input,shape=self.shape)


class L2Norm(UtilityLayer):
	
	def __init__(self,input,name):
		super(L2Norm,self).__init__(input,name)
		
	def setupOutput(self):
		with tf.variable_scope(self.name):
			self.output = tf.nn.l2_normalize(self.input,-1)

			
class Resample(UtilityLayer):
	
	def __init__(self,input,outputShape,name,method=tf.image.ResizeMethod.BICUBIC,alignCorners=True):
		self.outputShape = outputShape
		self.method = method
		self.alignCorners = alignCorners
		super(Resample,self).__init__(input,name)
		
	def setupOutput(self):
		with tf.variable_scope(self.name):
			try:
				self.output = tf.image.resize_images(self.input,self.outputShape,method=self.method)#,align_corners=self.alignCorners)
			except:
				self.output = tf.image.resize_images(self.input,self.outputShape[0],self.outputShape[1],method=self.method)#,align_corners=self.alignCorners)


class Dropout(UtilityLayer):

	def __init__(self,input,name):
		self.input = input
		self.name = name
		super(Dropout,self).__init__(input,name)
		
	def initialize(self):
		with tf.variable_scope(self.name):
			self.keepProb = tf.placeholder('float')			# Variable to hold the dropout probability
		
	def setupOutput(self):
		with tf.variable_scope(self.name):
			self.output = tf.nn.dropout(self.input,self.keepProb)
			self.output.get_shape = self.input.get_shape		# DEBUG: remove this whenever TensorFlow fixes this bug



#*** Main Part ***
if __name__ == '__main__':
	import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	session = tf.InteractiveSession()

	x = tf.placeholder('float',shape=[None,784],name='input')		# Input tensor
	y_ = tf.placeholder('float', shape=[None,10],name='correctLabels') 		# Correct labels
	trainingIterations = 5000

#	L1 = ReLu(x,512,'relu1')
#	L2 = ReLu(L1.output,128,'relu2')
#	L3 = ReLu(L2.output,64,'relu3')
#	L4 = SoftMax(x,10,'softmax')
#	y = L4.output
#	trainDict = {}; testDict = trainDict
#	logName = 'logs/softmax'


	xImage = tf.reshape(x,[-1,28,28,1])		# Reshape samples to 28x28x1 images
	L1 = Conv2D(xImage,[5,5,1,32],'Conv1')
	L2 = MaxPool2x2(L1.output,'MaxPool1')
	L3 = Conv2D(L2.output,[5,5,32,64],'Conv2')
	L4 = MaxPool2x2(L3.output,'MaxPool2')
	L5 = ReLu(L4.output,128,'relu1')
	L6 = Dropout(L5.output,'dropout')
	L7 = SoftMax(L5.output,10,'softmax')
	y = L7.output
	kp = 0.5; trainDict = {L6.keepProb:kp}
	kp = 1.0; testDict = {L6.keepProb:kp}
	logName = 'logs/Conv'

	# Training and evaluation
	crossEntropy = -tf.reduce_sum(y_*tf.log(y))		# cost function
	trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)
	trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
	correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
	train(session=session,trainingData=mnist.train,testingData=mnist.test,truth=y_,input=x,cost=crossEntropy,trainingStep=trainStep,accuracy=accuracy,iterations=trainingIterations,miniBatch=100,trainDict=trainDict,testDict=testDict,logName=logName)

	#plotFields(L1,[28,28],figOffset=1)

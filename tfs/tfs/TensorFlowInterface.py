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
import pylab
import numpy
import time


class Base(object):
	
	def __init__(self,name):
		self.name = name
		self.setup()
	
	def setup(self):
		pass
	
	def output(self):
		return None


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






def weightVariable(shape,std=1.0,name=None):
	# Create a set of weights initialized with truncated normal random values
	name = 'weights' if name is None else name
	return tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=std/math.sqrt(shape[0])))

def biasVariable(shape,bias=0.1,name=None):
	# create a set of bias nodes initialized with a constant 0.1
	name = 'biases' if name is None else name
	return tf.get_variable(name,shape,initializer=tf.constant_initializer(bias))

def conv2d(x,W,strides=[1,1,1,1],name=None):
	# return an op that convolves x with W
	strides = numpy.array(strides)
	if strides.size == 1:
		strides = numpy.array([1,strides,strides,1])
	elif strides.size == 2:
		strides = numpy.array([1,strides[0],strides[1],1])
	if numpy.any(strides < 1):
		strides = numpy.around(1./strides).astype(numpy.uint8)
		return tf.nn.conv2d_transpose(x,W,strides=strides.tolist(),padding='SAME',name=name)
	else:
		return tf.nn.conv2d(x,W,strides=strides.tolist(),padding='SAME',name=name)
	

def conv3d(x,W,strides=1,name=None):
	# return an op that convolves x with W
	strides = numpy.array(strides)
	if strides.size == 1:
		strides = numpy.array([1,strides,strides,strides[0],1])
	elif strides.size == 3:
		strides = numpy.array([1,strides[0],strides[1],strides[2],1])
	if numpy.any(strides < 1):
		strides = numpy.around(1./strides).astype(numpy.uint8)
		return tf.nn.conv3d_transpose(x,W,strides=strides.tolist(),padding='SAME',name=name)
	else:
		return tf.nn.conv3d(x,W,strides=strides.tolist(),padding='SAME',name=name)

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
	try:
		W = layer.W
	except:
		W = layer
	wp = W.eval().transpose();
	if len(numpy.shape(wp)) < 4:		# Fully connected layer, has no shape
		fields = numpy.reshape(wp,list(wp.shape[0:-1])+fieldShape)	
	else:			# Convolutional layer already has shape
		features, channels, iy, ix = numpy.shape(wp)
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = numpy.reshape(wp,[features*channels,iy,ix])

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))

	fig = pylab.figure(figOffset); pylab.clf()
	
	# Using image grid
	from mpl_toolkits.axes_grid1 import ImageGrid
	grid = ImageGrid(fig,111,nrows_ncols=(perRow,perColumn),axes_pad=padding,cbar_mode='single')
	for i in range(0,numpy.shape(fields)[0]):
		im = grid[i].imshow(fields[i],cmap=cmap); 

	grid.cbar_axes[0].colorbar(im)
	pylab.title('%s Receptive Fields' % layer.name)
	
	# old way
	# fields2 = numpy.vstack([fields,numpy.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])
	# tiled = []
	# for i in range(0,perColumn*perRow,perColumn):
	# 	tiled.append(numpy.hstack(fields2[i:i+perColumn]))
	# 
	# tiled = numpy.vstack(tiled)
	# pylab.figure(figOffset); pylab.clf(); pylab.imshow(tiled,cmap=cmap); pylab.title('%s Receptive Fields' % layer.name); pylab.colorbar();
	pylab.figure(figOffset+1); pylab.clf(); pylab.imshow(numpy.sum(numpy.abs(fields),0),cmap=cmap); pylab.title('%s Total Absolute Input Dependency' % layer.name); pylab.colorbar()



def plotOutput(layer,feed_dict,fieldShape=None,channel=None,figOffset=1,cmap=None):
	# Output summary
	try:
		W = layer.output
	except:
		W = layer
	wp = W.eval(feed_dict=feed_dict);
	if len(numpy.shape(wp)) < 4:		# Fully connected layer, has no shape
		temp = numpy.zeros(numpy.product(fieldShape)); temp[0:numpy.shape(wp.ravel())[0]] = wp.ravel()
		fields = numpy.reshape(temp,[1]+fieldShape)
	else:			# Convolutional layer already has shape
		wp = numpy.rollaxis(wp,3,0)
		features, channels, iy,ix = numpy.shape(wp)
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = numpy.reshape(wp,[features*channels,iy,ix])

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))
	fields2 = numpy.vstack([fields,numpy.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])
	tiled = []
	for i in range(0,perColumn*perRow,perColumn):
		tiled.append(numpy.hstack(fields2[i:i+perColumn]))

	tiled = numpy.vstack(tiled)
	if figOffset is not None:
		pylab.figure(figOffset); pylab.clf(); 

	pylab.imshow(tiled,cmap=cmap); pylab.title('%s Output' % layer.name); pylab.colorbar();



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
				summary,testAccuracy,testCost = session.run([mergedSummary,accuracy,cost],feed_dict=testDict)
				if logName is not None:
					writer.add_summary(summary,i)
			else:
				testAccuracy,testCost = session.run([accuracy,cost],feed_dict=testDict)[0]

			print('At batch {}: accuracy: {} cost: {} ({} samples/s)'.format(i,testAccuracy,testCost,(i-lastIterations)/(time.time()-lastTime)*miniBatch))
			lastTime = time.time(); lastIterations = i

		trainDict.update({input:batch[0],truth:batch[1]})
		trainingStep.run(feed_dict=trainDict)

	try:
		# Only works with mnist-type data object
		testDict.update({input:testingData.images, truth:testingData.labels})
		print('Test accuracy: {}'.format(accuracy.eval(feed_dict=testDict)))
	except:
		pass
	



from TensorFlowInterface import *
import input_data
from pylab import *
from numpy import *
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
session = tf.InteractiveSession()

# Hacky class to add a null option to the MNIST one hot vector (position 10)
class MNISTModifier(object):
	
	def __init__(self,data):
		self.data = data
	
	def next_batch(self,miniBatch):
		batch = list(self.data.next_batch(miniBatch))
		batch[1] = np.hstack((batch[1],zeros([shape(batch[1])[0]]+[1])))
		return batch

class Container(object):
	pass

temp = Container(); temp.train = MNISTModifier(mnist.train); temp.test = MNISTModifier(mnist.test)
mnist = temp

x = tf.placeholder('float',shape=[None,784],name='input')		# Input tensor
y_ = tf.placeholder('float', shape=[None,11],name='correctLabels') 		# Correct labels

xImage = tf.reshape(x,[-1,28,28,1])		# Reshape samples to 28x28x1 images
trainingIterations = 5000

# Standard conv net from Session 3
L1 = Conv2D(xImage,[5,5,1,32],'Conv1')
L2 = MaxPool2x2(L1.output,'MaxPool1')
L0do = Dropout(L2.output,'dropout')
L3 = Conv2D(L2.output,[5,5,32,64],'Conv2')
L4 = MaxPool2x2(L3.output,'MaxPool2')
L5 = ReLu(L4.output,128,'relu1')
L6 = SoftMax(L5.output,11,'softmax')
y = L6.output
kp = 0.5; trainDict = {L0do.keepProb:kp}
kp = 1.0; testDict = {L0do.keepProb:kp}
logName = None #logName = 'logs/Conv'


# Training and evaluation
crossEntropy = -tf.reduce_sum(y_*tf.log(y))		# cost function
trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)
trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
train(session=session,trainingData=mnist.train,testingData=mnist.test,truth=y_,input=x,cost=crossEntropy,trainingStep=trainStep,accuracy=accuracy,iterations=trainingIterations,miniBatch=100,trainDict=trainDict,testDict=testDict,logName=logName)




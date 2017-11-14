from tfs import *
from pylab import *
from numpy import *
import glob, os
import nibabel as nib

matplotlib.interactive(True)
session = tf.InteractiveSession()

dataPath = '/Volumes/PACS/data/corpusCallosum'



# Class to serve up segmented images
class CCData(object):
	
	def __init__(self,paths):
		self.paths = paths
	
	def getSlices(self,paths):
		image,truth = paths
		image = nib.load(image).get_data(); truth = nib.load(truth).get_data()
		slicesWithValues = [unique(s) for s in where(truth>0)]
		sliceAxis = argmin([len(s) for s in slicesWithValues])
		slicesWithValues = slicesWithValues[sliceAxis]
		slc = repeat(-1,3); slc[sliceAxis] = slicesWithValues[0]
		return (image[slc][0],truth[slc][0])
	
	def next_batch(self,miniBatch=None):
		if miniBatch is None or miniBatch==len(self.paths):
			batch = arange(0,len(self.paths))
		else:
			batch = random.choice(arange(0,len(self.paths)),miniBatch)
		images = [self.getSlices(self.paths[i]) for i in batch]
		return zip(*images)
		

class Container(object):

	def __init__(self,dataPath,reserve=2):
		self.dataPath = dataPath
		images = glob.glob(os.path.join(dataPath,'?????.nii.gz'))
		images = [(i,i.replace('.nii.gz','_cc.nii.gz')) for i in images]
		self.train = CCData(images[0:-reserve])
		self.test = CCData(images[reserve:])


data = Container(dataPath,reserve=2)
batch = data.train.next_batch(2)

trainingIterations = 1000

x = tf.placeholder('float',shape=[None,None,None],name='input')
y_ = tf.placeholder('float', shape=[None,None,None],name='truth')
y_OneHot = tf.one_hot(indices=tf.cast(y_,tf.int32),depth=2,name='truthOneHot')
xInput = tf.expand_dims(x,axis=3,name='xInput')

# Standard conv net from Session 3
net = L1 = tf.layers.conv2d(
					inputs=xInput,
					filters=10,
					kernel_size=[5,5],
					strides = 1,
					padding = 'same',
					activation=tf.nn.relu,
					name='conv1'
				)
net = L1 = tf.layers.conv2d(
					inputs=net,
					filters=20,
					kernel_size=[5,5],
					strides = 1,
					padding = 'same',
					activation=tf.nn.relu,
					name='conv2'
				)
net = L3 = tf.layers.conv2d(
					inputs=net,
					filters=2,
					kernel_size=[5,5],
					strides = 1,
					padding = 'same',
					activation=tf.nn.softmax,
					name='softmax'
				)

y = net

kp = 0.5; trainDict = {}#{L0do.keepProb:kp}
kp = 1.0; testDict = {}#{L0do.keepProb:kp}
logName = None #logName = 'logs/Conv'


# Training and evaluation
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_OneHot, logits=y)

trainStep = tf.train.AdamOptimizer(1e-4).minimize(loss)
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_OneHot,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
train(session=session,trainingData=data.train,testingData=data.test,truth=y_,input=x,cost=loss,trainingStep=trainStep,accuracy=accuracy,iterations=trainingIterations,miniBatch=2,trainDict=trainDict,testDict=testDict,logName=logName)



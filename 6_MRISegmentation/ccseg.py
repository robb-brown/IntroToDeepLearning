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


trainDict = {}
testDict = {}
logName = None #logName = 'logs/Conv'


# Training and evaluation
loss = None

trainStep = tf.train.AdamOptimizer(1e-4).minimize(loss)
correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_OneHot,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))
train(session=session,trainingData=data.train,testingData=data.test,truth=y_OneHot,input=x,cost=loss,trainingStep=trainStep,accuracy=accuracy,iterations=trainingIterations,miniBatch=2,trainDict=trainDict,testDict=testDict,logName=logName)



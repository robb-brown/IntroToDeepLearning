from tfs import *
from tfs import input_data
from pylab import *
from numpy import *
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
matplotlib.interactive(True)
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
	
	def __init__(self,mnist):
		self.train = MNISTModifier(mnist.train)
		self.test = MNISTModifier(mnist.test)
		
		
def makeBigImage(data,Ndigits=5,width=100,height=100,overlap=False,digitShape=[28,28],maxTries=1000):
	newImage = zeros((height,width),float32)
	truthImage = zeros(list(shape(newImage))+[11],float32)
	batch = data.next_batch(Ndigits)
	tries = 0
	while Ndigits > 0:
		y,x = np.random.randint(0,height-digitShape[0],2)
		if not np.any(truthImage[y:y+digitShape[0],x:x+digitShape[0]]) or overlap:
			newImage[y:y+digitShape[0],x:x+digitShape[0]] = batch[0][Ndigits-1].reshape(digitShape)
			truthImage[y:y+digitShape[0],x:x+digitShape[0]] = batch[1][Ndigits-1]
			Ndigits -= 1; tries = 0
		else:
			tries += 1
		if tries > maxTries:
			break
	truthImage[:,:,10] += np.all(truthImage<0.5,axis=-1)
	return (newImage,truthImage)
		


# Hack our MNIST dataset so it has a null option
mnist = Container(mnist)

# x = tf.placeholder('float',shape=[None,784],name='input')		# Input tensor
# y_ = tf.placeholder('float', shape=[None,11],name='correctLabels') 		# Correct labels

trainingIterations = 1000
height = width = 100
x2 = tf.placeholder('float',shape=[None,height,width],name='input2')		# Input tensor; we shouldn't need to specify dimensions except TensorFlow....
y_2 = tf.placeholder('float', shape=[None,height,width,11],name='correctLabels2') # Correct labels


inLayer = tf.expand_dims(x2,-1)
L1 = Conv2D(inLayer,[5,5,1,32],'Conv1')
L2 = MaxPool2x2(L1.output,'MaxPool1')
L3 = Conv2D(L2.output,[5,5,32,64],'Conv2')
L4 = MaxPool2x2(L3.output,'MaxPool2')
L5 = Conv2D(L4.output,[7,7,64,128],'relu1-conv')
L6 = ConvSoftMax(L5.output,[1,1,128,11],'softmax-conv')
#L7 = Resample(L6.output,L1.input.get_shape().as_list()[1:3],'upsample')
y = L6.output

trainDict = {}
testDict = {}


truthDownsampleLayer = Resample(y_2,L6.output.get_shape().as_list()[1:3],'ydownsample')
y2 = L6.output; y_2b = truthDownsampleLayer.output

crossEntropy = -tf.reduce_sum(y_2b*tf.log(y2))		# cost function
trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(y2,3), tf.argmax(y_2b,3))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))

print("Training Image recognizer")
iterations = trainingIterations; batchSize = 2
lastTime = 0; lastIterations = 0; trainingStep=trainStep; accuracy=accuracy;
tf.global_variables_initializer().run()

for i in range(iterations):						# Do some more training
	# batchIndices = random.choice(range(0,len(trainingData)),batchSize)
	# batch = [trainingData[batchIndices],truthData[batchIndices]]
	batch = list(zip(*[makeBigImage(data=mnist.train,Ndigits=5,width=width,height=height,overlap=False) for j in range(0,3)]))
	if (i%100 == 0) or (time.time()-lastTime > 5):
		testDict.update({x2:batch[0],y_2:batch[1]})
		testAccuracy = session.run([accuracy],feed_dict=testDict)[0]
		print('Accuracy at batch {}: {} ({} samples/s)'.format(i,testAccuracy,(i-lastIterations)/(time.time()-lastTime)*batchSize))
		lastTime = time.time(); lastIterations = i
	
	trainDict.update({x2:batch[0],y_2:batch[1]})
	trainingStep.run(feed_dict=trainDict)

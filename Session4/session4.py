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



# Now turn our trained conv net into a fully convolutional network by replacing the fully
# connected layers (L5 and L6) with convoluational layers
# Note that I'm copying the L5 and L6 weights and biases into the new layers

L5fc = Conv2D(L4.output,[7,7,64,128],'relu1-conv')
L5fc.W = tf.reshape(L5.W,[7,7,64,128])
L5fc.b = L5.b
L5fc.setupOutput()

L6fc = ConvSoftMax(L5fc.output,[1,1,128,11],'softmax-conv')
L6fc.W = tf.reshape(L6.W,[1,1,128,11])
L6fc.b = L6.b
L6fc.setupOutput()

# This last layer upsamples the output back up to the input size for comparison
L7 = Resample(L6fc.output,L1.input.get_shape().as_list()[1:3],'upsample')



# Choose an example and show it for comparison
example = mnist.test.next_batch(1); image = reshape(example[0][0],(28,28))
feed_dict = {x:example[0],y_:example[1],L0do.keepProb:1.0}

# Regular convolutional network
figure(1); clf(); subplot(231); imshow(image,cmap=cm.gray); title('Correct is %d' % where(example[1]>0)[1][0])
subplot(232); plotOutput(L1,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(233); plotOutput(L2,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(235); plotOutput(L3,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(236); plotOutput(L4,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(234); plotOutput(L6,fieldShape=[11,1],feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);

# Fully convolutional network
figure(2); clf(); subplot(121); imshow(np.argmax(L7.output.eval(feed_dict=feed_dict),axis=-1)[0],vmin=0,vmax=10); colorbar(); title('Digit Identification')
contour(image,[0.5],colors=['k'])
subplot(122); imshow(np.max(L7.output.eval(feed_dict=feed_dict),axis=-1)[0],cmap=cm.gray); colorbar(); title('Strength of Identification')
contour(image,[0.5],colors=['b'])



# Do a bit more training but stick a null image in each batch to teach the network about
# places where there is no value.  This code copied from TensorFlowInterface
nullImage = [zeros(28*28,float32)]; nullTruth = array([zeros(11,float32)]); nullTruth[:,0] = 1.
trainingData = mnist.train; iterations = trainingIterations; miniBatch = 100
lastTime = 0; lastIterations = 0; trainingStep=trainStep; accuracy=accuracy
for i in range(iterations):						# Do some more training
	batch = list(trainingData.next_batch(miniBatch))
	batch[0] = np.concatenate((batch[0],nullImage)); batch[1] = concatenate((batch[1],nullTruth))
	if (i%100 == 0) or (time.time()-lastTime > 5):
		testDict.update({x:batch[0],y_:batch[1]})
		testAccuracy = session.run([accuracy],feed_dict=testDict)[0]
		print 'Accuracy at batch %d: %g (%g samples/s)' % (i,testAccuracy,(i-lastIterations)/(time.time()-lastTime)*miniBatch)
		lastTime = time.time(); lastIterations = i
	
	trainDict.update({x:batch[0],y_:batch[1]})
	trainingStep.run(feed_dict=trainDict)


# Plot again and see the difference
figure(3); clf(); subplot(121); imshow(np.argmax(L7.output.eval(feed_dict=feed_dict),axis=-1)[0],vmin=0,vmax=10); colorbar(); title('Digit Identification')
contour(image,[0.5],colors=['k'])
subplot(122); imshow(np.max(L7.output.eval(feed_dict=feed_dict),axis=-1)[0],cmap=cm.gray); colorbar(); title('Strength of Identification')
contour(image,[0.5],colors=['b'])



# Make a bigger image with several numbers in it and see if we can find their location
# AND identify them!

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

height = width = 100
newImage,truthImage = makeBigImage(data = mnist.train,Ndigits=5,width=width,height=height,overlap=False)

# What's it look like?
figure(4); clf(); subplot(121); imshow(newImage); subplot(122); imshow(np.argmax(truthImage,axis=-1),vmin=0,vmax=10)

# We need new placeholders to hold the bigger image
x2 = tf.placeholder('float',shape=[None,height,width],name='input2')		# Input tensor; we shouldn't need to specify dimensions except TensorFlow....
y_2 = tf.placeholder('float', shape=[None,height,width,11],name='correctLabels2') # Correct labels

# We now need to rebuild our network to work with the new input
lastLayerOutput = tf.expand_dims(x2,-1)			# Add the extra dummy dimension
for layer in [L1,L2,L3,L4,L5fc,L6fc]:
	layer.input = lastLayerOutput; 
	layer.setupOutput()
	lastLayerOutput = layer.output

# Need to make a new L7 to match the input
L7 = Resample(L6fc.output,L1.input.get_shape().as_list()[1:3],'upsample2')



# Forward prop
feed_dict = {x2:[newImage],y_2:[truthImage],L0do.keepProb:1.0}
figure(5); clf(); subplot(121); imshow(np.argmax(L7.output.eval(feed_dict=feed_dict),axis=-1)[0],vmin=0,vmax=10); colorbar(); title('Digit Identification')
contour(newImage,[0.5],colors=['k'])
subplot(122); imshow(np.max(L7.output.eval(feed_dict=feed_dict),axis=-1)[0],cmap=cm.gray); colorbar(); title('Strength of Identification')
contour(newImage,[0.5],colors=['b'])


# Train again, on the big image specifically
# Tensorflow doesn't know how to do gradients on interpolated images (!) so lets downsample
# the truth to match the L6 output

truthDownsampleLayer = Resample(y_2,L6fc.output.get_shape().as_list()[1:3],'ydownsample')

y2 = L6fc.output; y_2b = truthDownsampleLayer.output
crossEntropy = -tf.reduce_sum(y_2b*tf.log(y2))		# cost function
trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(y2,3), tf.argmax(y_2b,3))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,'float'))

tf.initialize_all_variables().run()

print "Training Image recognizer"
iterations = trainingIterations; batchSize = 2
lastTime = 0; lastIterations = 0; trainingStep=trainStep; accuracy=accuracy;
for i in range(iterations):						# Do some more training
	# batchIndices = random.choice(range(0,len(trainingData)),batchSize)
	# batch = [trainingData[batchIndices],truthData[batchIndices]]
	batch = zip(*[makeBigImage(data=mnist.train,Ndigits=5,width=width,height=height,overlap=False) for j in range(0,3)])
	if (i%100 == 0) or (time.time()-lastTime > 5):
		testDict.update({x2:batch[0],y_2:batch[1]})
		testAccuracy = session.run([accuracy],feed_dict=testDict)[0]
		print 'Accuracy at batch %d: %g (%g samples/s)' % (i,testAccuracy,(i-lastIterations)/(time.time()-lastTime)*batchSize)
		lastTime = time.time(); lastIterations = i
	
	trainDict.update({x2:batch[0],y_2:batch[1]})
	trainingStep.run(feed_dict=trainDict)


feed_dict = {x2:[newImage],y_2:[truthImage],L0do.keepProb:1.0}
classification = np.argmax(L7.output.eval(feed_dict=feed_dict),axis=-1)[0]
confidence = np.max(L7.output.eval(feed_dict=feed_dict),axis=-1)[0]
figure(6); clf(); subplot(121); imshow(classification,vmin=0,vmax=10); colorbar(); title('Digit Identification')
contour(newImage,[0.5],colors=['k'])
subplot(122); imshow(confidence,cmap=cm.gray); colorbar(); title('Strength of Identification')
contour(newImage,[0.5],colors=['b'])

figure(7); clf(); imshow(confidence*10,cmap=cm.gray); imshow(classification,vmin=0,vmax=10,alpha=0.3)
contour(newImage,[0.5],colors=['k'])

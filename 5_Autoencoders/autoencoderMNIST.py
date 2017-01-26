from TensorFlowInterface import *
import input_data
from pylab import *
from numpy import *
import copy

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
matplotlib.interactive(True)
session = tf.InteractiveSession()

x = tf.placeholder('float',shape=[None,784],name='input')		# Input tensor
y_ = tf.placeholder('float', shape=[None,10],name='correctLabels') 		# Correct labels



# ---------------------   Fully connected network with dropout  ---------------------

trainingIterations = -8
yp = x
inputDO = Dropout(x,'inputDropout')

layerPlan = [100,50]
layerPlan = [500,300,100];

layers = dict(encoder=[],decoder=[]); inLayer = inputDO
for l,nUnits in enumerate(layerPlan):
	layer = ReLu(inLayer.output,nUnits,'encoder%d'%l)
	dropout = Dropout(layer.output,'encoderDO%d'%l)
	layers['encoder'].append([layer,dropout]); inLayer = dropout

for l,nUnits in enumerate(layerPlan[::-1]):
	layer = ReLu(inLayer.output,nUnits,'decoder%d'%l)
	dropout = Dropout(layer.output,'decoderDO%d'%l)
	layers['decoder'].append([layer,dropout]); inLayer = dropout

layer = ReLu(inLayer.output,784,'decoder%d'%(l+1))
layers['decoder'].append([layer,None])
y = layer.output

inputKP = 0.7; hiddenKP = 0.3
#inputKP = 1.0; hiddenKP = 1.0
trainDict = dict([(layer[1].keepProb,inputKP) for layer in layers['encoder']])
trainDict.update(dict([(layer[1].keepProb,inputKP) for layer in layers['decoder'] if layer[1] is not None]))
trainDict[inputDO.keepProb] = hiddenKP

testDict = dict([(layer[1].keepProb,1.0) for layer in layers['encoder']])
testDict.update(dict([(layer[1].keepProb,1.0) for layer in layers['decoder'] if layer[1] is not None]))
testDict[inputDO.keepProb] = 1.0

logName = 'logs/autoencoderFullyConnected'

# ----------------------------------------------------------------------------------

# ---------------------   Convnet with dropout  ---------------------
if 0:
	trainingIterations = 1000
	xImage = tf.reshape(x,[-1,28,28,1])		# Reshape samples to 28x28x1 images
	yp = xImage

	inputDO = Dropout(xImage,'inputDropout')
	layerPlan = [25,9]; layers = dict(encoder=[],decoder=[]); inLayer = inputDO; lastUnits = 1
	for l,nUnits in enumerate(layerPlan):
		layer = Conv2D(inLayer.output,[5,5,lastUnits,nUnits],'encoder%d'%l)
		dropout = Dropout(layer.output,'encoderDO%d'%l)
		layers['encoder'].append([layer,dropout]); inLayer = dropout; lastUnits = nUnits

	for l,nUnits in enumerate(layerPlan[::-1]):
		layer = Conv2D(inLayer.output,[5,5,lastUnits,nUnits],'decoder%d'%l)
		dropout = Dropout(layer.output,'decoderDO%d'%l)
		layers['decoder'].append([layer,dropout]); inLayer = dropout; lastUnits = nUnits

	layer = Conv2D(inLayer.output,[5,5,lastUnits,nUnits],'decoder%d'%(l+1))
	layers['decoder'].append([layer,None])
	y = layer.output

	inputKP = 0.7; hiddenKP = 0.3
	#inputKP = 1.0; hiddenKP = 1.0
	trainDict = dict([(layer[1].keepProb,inputKP) for layer in layers['encoder']])
	trainDict.update(dict([(layer[1].keepProb,inputKP) for layer in layers['decoder'] if layer[1] is not None]))
	trainDict[inputDO.keepProb] = hiddenKP

	testDict = dict([(layer[1].keepProb,1.0) for layer in layers['encoder']])
	testDict.update(dict([(layer[1].keepProb,1.0) for layer in layers['decoder'] if layer[1] is not None]))
	testDict[inputDO.keepProb] = 1.0

	logName = 'logs/autoencoderFullyConnected'


# Standard conv net
# L1 = Conv2D(xImage,[5,5,1,32],'Conv1')
# L2 = MaxPool2x2(L1.output,'MaxPool1')
# L0do = Dropout(L2.output,'dropout')
# L3 = Conv2D(L2.output,[5,5,32,64],'Conv2')
# L4 = MaxPool2x2(L3.output,'MaxPool2')
# L5 = ReLu(L4.output,128,'relu1')
# L6 = SoftMax(L5.output,10,'softmax')
# y = L6.output
# kp = 0.5; trainDict = {L0do.keepProb:kp}
# kp = 1.0; testDict = {L0do.keepProb:kp}
# logName = None #logName = 'logs/Conv'



# Training and evaluation

# Mean squared error between the input and reconstruction
cost = tf.sqrt(tf.reduce_mean(tf.square(yp-y)))

trainStep = tf.train.AdamOptimizer(1e-4).minimize(cost)
train(session=session,trainingData=mnist.train,testingData=mnist.test,truth=y_,input=x,cost=cost,trainingStep=trainStep,iterations=trainingIterations,miniBatch=100,trainDict=trainDict,testDict=testDict,logName=logName)


# Mini viewer
plotFields(layers['encoder'][0][0],fieldShape=[28,28],maxFields=100,cmap=cm.coolwarm)



example = mnist.test.next_batch(1); image = reshape(example[0][0],(28,28))
testDict.update({x:example[0],y_:example[1]})

for l in range(len(layers['encoder'])):
	figure('Layer Output'); clf()
	layer = layers['encoder'][l][0]
	fieldShape = [ceil(sqrt(layer.units)),ceil(layer.units/ceil(sqrt(layer.units)))]
	plotOutput(layer,feed_dict=testDict,fieldShape=fieldShape,cmap=cm.inferno,figOffset=None);
	savefig('test/layer%dOutput.pdf' % l,transparent=True)

figure('digitOriginal'); clf();
imshow(example[0].reshape((28,28)),cmap=cm.gray)
savefig('test/digitOriginal.jpg')
recon = session.run(y,feed_dict=testDict)
figure('digitReconstructed'); clf();
imshow(recon.reshape((28,28)),cmap=cm.gray)
savefig('test/digitOriginalRecon.jpg')

snr = 5.1
image = copy.deepcopy(example[0]);
image += random.normal(0,1./snr,shape(image))
figure('noisy image'); clf();
imshow(image.reshape((28,28)),cmap=cm.gray)
savefig('test/digitSNR%0.1f.jpg'%snr)

testDict.update({x:image})
recon = session.run(y,feed_dict=testDict)
figure('reconstruction'); clf();
imshow(recon.reshape((28,28)),cmap=cm.gray)
savefig('test/digitSNR%dRecon.jpg'%snr)

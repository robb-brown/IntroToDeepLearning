from TensorFlowInterface import *
import input_data
from pylab import *
from numpy import *
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
matplotlib.interactive(True)
session = tf.InteractiveSession()

x = tf.placeholder('float',shape=[None,784],name='input')		# Input tensor
y_ = tf.placeholder('float', shape=[None,10],name='correctLabels') 		# Correct labels

# A four layer fully connected net for comparison
# trainingIterations = 10000
# L1 = ReLu(x,512,'relu1')
# L2 = ReLu(L1.output,128,'relu2')
# L3 = ReLu(L2.output,64,'relu3')
# L4 = SoftMax(x,10,'softmax')
# y = L4.output
# trainDict = {}; testDict = trainDict
## logName = 'logs/4LayerFullyConnected'


xImage = tf.reshape(x,[-1,28,28,1])		# Reshape samples to 28x28x1 images
trainingIterations = 1000

# Standard conv net
L1 = Conv2D(xImage,[5,5,1,32],'Conv1')
L2 = MaxPool2x2(L1.output,'MaxPool1')
L0do = Dropout(L2.output,'dropout')
L3 = Conv2D(L2.output,[5,5,32,64],'Conv2')
L4 = MaxPool2x2(L3.output,'MaxPool2')
L5 = ReLu(L4.output,128,'relu1')
L6 = SoftMax(L5.output,10,'softmax')
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

# Mini viewer
plotFields(L1,cmap=cm.coolwarm)
example = mnist.test.next_batch(1); image = reshape(example[0][0],(28,28))
feed_dict = {x:example[0],y_:example[1],L0do.keepProb:1.0}
figure(1); clf(); subplot(231); imshow(image,cmap=cm.gray); title('Correct is %d' % where(example[1]>0)[1][0])
subplot(232); plotOutput(L1,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(233); plotOutput(L2,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(235); plotOutput(L3,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(236); plotOutput(L4,feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);
subplot(234); plotOutput(L6,fieldShape=[10,1],feed_dict=feed_dict,cmap=cm.inferno,figOffset=None);

# Individual interesting fields
# plotOutput(L1,feed_dict={x:example[0],y_:example[1]},cmap=cm.inferno); figure(2); clf(); imshow(image,cmap=cm.gray)
# plotOutput(L3,fieldShape=[11,12],feed_dict=feed_dict,cmap=cm.inferno); figure(2); clf(); imshow(image,cmap=cm.gray)
# plotOutput(L6,fieldShape=[10,1],feed_dict=feed_dict,cmap=cm.inferno); figure(2); clf(); imshow(image,cmap=cm.gray)



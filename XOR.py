
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
print (mnist.data.shape)
nperclass = 36
classes = [30,30 ,30]
        #classes = [10, 13, 28] # A, C, S
nclasses = len(classes)
print(len(classes))
import numpy as np
import tensorflow as tf 
 

	# Read in the data and prepare it
data = sio.loadmat('binaryalphadigs.mat')
i=0,j=0
for i in range(36):
    for j in range (32):
    y=data['dat'][i][j]
    



(x_train, y_train),(x_test, y_test) = data.load_data()
train_images=data['dat'][:int(len(data['dat'])*0.8)]
train_labels =data['classlabels'][:int(len(data['classlabels'])*0.8)]
test_images=data['dat'][int(len(data['dat'])*0.8):]
test_labels=data['classlabels'][:int(len(data['classlabels'])*0.8):]
x=data['classlabels']
print(x)
y=data['dat'][2][1]
print(y)
plt.imshow(data['dat'][2][1])

'''mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded'''
n_input = 260   # input layer (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 36   # output layer (0-9 digits)
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5        
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
print(Y)
from sklearn.model_selection import train_test_split 
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
train_labels, test_size=0.25, random_state=42) 
keep_prob = tf.placeholder(tf.float32)
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(n_iterations):
    batch_x, batch_y = .next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))
'''inputs = np.ones((nclasses, nperclass, 20*16))
labels = np.zeros((nclasses, nperclass, nclasses))
for k in range(nclasses):
   for m in range(nperclass):
    inputs[k,m,:] = (data['dat'][classes[k],m].ravel()).astype('float')
    labels[k,m,k] = 1.
nexamples = 20
v = inputs[:,:nexamples,:].reshape(nclasses*nexamples, 20*16)
l = labels[:,:nexamples,:].reshape(nclasses*nexamples, nclasses)
import pylab as pl
pl.figure() 
for i in range(60):
 pl.subplot(6,10,i+1), pl.imshow(v[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')
'''
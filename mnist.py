import tensorflow as tf
import argparse

#need to import data here
#mnist data loaded here, change!
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

##########
parser = argparse.ArgumentParser(description='Demonstrate user input argument parsing')
# collects one argument
parser.add_argument('numLayers', type=int, help='Number of hidden layers')

# collects argument/s in list (python version of arraylist)
parser.add_argument('numNodesPer', type=int, help='Number of nodes in each layer (If only 1 arg given, then 1 arg applied to each layer)', nargs='+')

args = parser.parse_args()
# if a user inputted more than 1 value for numNodesPer argument
if len(args.numNodesPer) > 1:
	# check to make sure there are same number of values as the number of layers
	if not len(args.numNodesPer) == args.numLayers:
		print 'Must have at least ' + str(args.numLayers) + ' of Nodes Per Layer arguments'
	n_nodes_hl = args.numNodesPer

if len(args.numNodesPer) == 1:
	n_nodes_hl = []
	numnodes = args.numNodesPer
	for i in xrange(args.numLayers):
		n_nodes_hl.extend(numnodes)

hidden_layers = [None]*args.numLayers

n_classes = 10 		 ###mnist data is handwritten digits 0-9
n_inputs = 784 		 ###mnist data is 28x28 pixels which we squahs to 784
batch_size = 500	 ### batch size can be tweaked depending on amount of RAM
x = tf.placeholder('float',[None, n_inputs])
y = tf.placeholder('float',[None, n_classes])
W = tf.Variable(tf.zeros([n_inputs,n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

#### basic function definitions for initializing weight and bias tensors
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#### basic function definitions for convolution and pooling layers
#### Note: further tweaking can be done here
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def neural_network_model(data):
	# must explicitely instantiate list in python if you want to insert/append
	# alternatively you can only insert into a new list with W_fc = [blah]
	W_fc = []
	b_fc = []
	l = []
	print n_nodes_hl
	for i in xrange(args.numLayers):
		if i==0:
			W_fc.insert(i, weight_variable([n_inputs,n_nodes_hl[i]]))
			b_fc.insert(i, bias_variable([n_nodes_hl[i]]))
		elif i == args.numLayers - 1:
			W_fc.insert(i, weight_variable([n_nodes_hl[i-1],n_classes]))
			b_fc.insert(i, bias_variable([n_classes]))
		else:
			W_fc.insert(i, weight_variable([n_nodes_hl[i-1],n_nodes_hl[i]]))
			b_fc.insert(i, bias_variable([n_nodes_hl[i]]))
 
	for i in xrange(args.numLayers):
		if i==0:
			l.insert(i, tf.nn.relu(tf.matmul(data,W_fc[i]) + b_fc[i]))
		elif i == args.numLayers - 1:
			l.insert(i, tf.matmul(l[i-1], W_fc[i]) + b_fc[i])
			return l[i]
		else:
			l.insert(i, tf.nn.relu(tf.matmul(l[i-1],W_fc[i]) + b_fc[i]))


#### Convolutional neural network with 2 layers of convolution and 2 layers of pooling (alternated)
#### followed by one fully connected layer of 1024 neruons
def conv_network_model(x):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

###removed because I dont fully understand
#    keep_prob = tf.placeholder(tf.float32)
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    output=tf.matmul(h_fc1, W_fc2) + b_fc2
    return output

#### training model from the youtube channel posted in pilot.wright.edu
#### youtube user sentdex
#### has been altered slightly to allow for different neural network models and to be interchangeable with below training model
def train_neural_network(x, network_model):
	prediction = network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	train_step = tf.train.AdamOptimizer().minimize(cost)
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		hm_epochs = 10
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([train_step, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#### training model from tensorflow deep mnist for experts walkthrough
#### has been altered slightly to allow for different neural network models and to be interchangeable with above training model
def train_neural_network2(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
		print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y: batch[1]})
		print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
		#train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
		#train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
		#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

#### time to train our network
train_neural_network(x, neural_network_model)

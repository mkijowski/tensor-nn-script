#going to do this very basic: import makes trensorflow libraries available
#for use. see https://docs.python.org/3/reference/import.html
import tensorflow as tf

#ibraries taht download mnist data sets to /tmp/data
from tensorflow.examples.tutorials.mnist import input_data
#reads the input data as one_hot (need more data on one_hot encoding)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#int values for number of nodes in each of the three hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

##mnist dependent data below (inputs and classes) change for other data sets
n_classes = 10 		 ###mnist data is handwritten digits 0-9
n_inputs = 784 		 ###mnist data is 28x28 pixels which we squahs to 784
batch_size = 500	 ### batch size can be tweaked depending on amount of RAM

## x is your input data and y is your output, you do not need to specify the size of the matrix [None, n_inputs], 
## but this will cause tensorflow to throw an error if data is loaded outside of that shape
## see https://www.tensorflow.org/versions/r0.10/api_docs/python/io_ops.html#placeholder
x = tf.placeholder('float',[None, n_inputs])
y = tf.placeholder('float',[None, n_classes])

<<<<<<< HEAD
=======
##see: https://www.tensorflow.org/versions/r0.10/api_docs/python/state_ops.html#Variable
## in order to use the variable swe first initialise them with zeros with the tf.zeros function 
## which requires the tensor's shape.  W is the variable for your weight tensor and b is the bias tensor
##These have been deprecated by the functions weight variable and bias_variable below
#W = tf.Variable(tf.zeros([n_inputs,n_classes]))
#b = tf.Variable(tf.zeros([n_classes]))

>>>>>>> 2599c0ed58c7b3bced08886cedeb0727e58d3f73
## function that takes input of the shape of a tensor and initializes it using truncated_normal distribution
##see: https://www.tensorflow.org/versions/r0.10/api_docs/python/state_ops.html#Variable
##and: https://www.tensorflow.org/versions/r0.10/api_docs/python/constant_op.html
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

## function that takes input of the shape of a tensor and initializes it to a constant
## see tf.variable link above
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

## function that controls the convolutional layers in the convolutional network model
## see: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#conv2d
## takes as input the OUTPUT DATA FROM PREV LAYER*reshaped* input image data x and a weight tensor W
##   has several more options see link above, but defined here are the stride distance and padding
<<<<<<< HEAD
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#The strides argument specifies how much the window shifts in each of the input dimensions. The dimensions are batch, height, width, chanels. Here, because the image is greyscale, there is only 1 dimension. A batch size of 1 means that a single example is processed at a time. A value of 1 in row/column means that we apply the operator at every row and column. When the padding parameter takes the value of 'SAME', this means that for those elements which have the filter sticking out during convolution, the portions of the filter that stick out are assumed to contribute 0 to the dot product. The size of the image remains the same after the operation remains the same. When the padding = 'VALID' option is used, the filter is centered along the center of the image, so that the filter does not stick out. So the output image size would have been 3X3. 
=======
##  ??? the dimension of the strides must be the same as the input, so why is it 4D and not 2D?
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
>>>>>>> 2599c0ed58c7b3bced08886cedeb0727e58d3f73

## function that controls the pooling layers in the convolutional network model
## see: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#max_pool
## takes as input the output of the previous layer x, and output the output of a 2x2 max pool
## ksize is the window size of the input tensor to max pool, strides is the size of the step that the window takes
## ???If these are the same this means there is not overlap in the pooling layer correct???
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

## function that defines a fully connected neural network model
## takes as input the input data "data"
def neural_network_model(data):
    ##creates weight and bias tensors for each fully connected layer, and for the output layer
    ## uses the functions weight_variable and bias_variable defined above.  Sends shape data as inputs
    W_fc1 = weight_variable([n_inputs, n_nodes_hl1])
    b_fc1 = bias_variable([n_nodes_hl1])

    W_fc2 = weight_variable([n_nodes_hl1, n_nodes_hl2])
    b_fc2 = bias_variable([n_nodes_hl2])

    W_fc3 = weight_variable([n_nodes_hl2, n_nodes_hl3])
    b_fc3 = bias_variable([n_nodes_hl3])

    W_output = weight_variable([n_nodes_hl3,n_classes])
    b_output = bias_variable([n_classes])

    ## the actual computation of the models layers including output layer.  performs basic matrix multiplacation of inputs x weights to each layer
    ## and adds the bias tensor
    ## uses rectilinear unit activation function
    ## see: https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#relu
    l1 = tf.nn.relu(tf.matmul(data,W_fc1) + b_fc1)
    l2 = tf.nn.relu(tf.matmul(l1,W_fc2) + b_fc2)
    l3 = tf.nn.relu(tf.matmul(l2,W_fc3) + b_fc3)
    output = tf.matmul(l3, W_output) + b_output

    ## returns the output of the output layer
    ## ??note this is basic output of that layer prior to using any activation function??
    return output

## function that defines a convolutional neural network model
## takes as input the input data "x"
## this was mostly modeled from: https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
## with many of the variables changed to fit with the above fully connected neural network model
def conv_network_model(x):
    ## creates a weight tensor of the given shape
    ## the shape corresponds to the convolutional layers filter size (5x5), the input data dimensionality (1D input of 784 pixels)
    ##  and number of features or parallel slices (32)
<<<<<<< HEAD
=======
    ## ?? why not omit the dimensionality data
>>>>>>> 2599c0ed58c7b3bced08886cedeb0727e58d3f73
    ## number of filters should be a power of 2 because this improves performance (look into this!!!!
    W_conv1 = weight_variable([5, 5, 1, 32])
    ## creates a bias tensor of the given shape
    b_conv1 = bias_variable([32])

    ## reshape the input to a 4d tensor
    ## takes input data x, a 1 dimensional, 784 pixel representation of the 28x28 image
    ## 2nd and 3rd dimension correspond to the image width and height, with the last dimension being the number of color channels (1)
    ## ??what is the first dimension??
    ## -1 is a special character that infers the size based on other data
    ## see https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#reshape
    x_image = tf.reshape(x, [-1,28,28,1])

    ## computes the output of the first convolutional layer with rectilinear unit activation
    ## uses convolution function above
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    ## applies max pooling to the output of the first convolutional layer
    ## uses pooling function above
    h_pool1 = max_pool_2x2(h_conv1)

    ## creates the weight tensor for the second layer of the given shape
    ## still suing filter size of 5x5, dimensionality of previous layer is 32, and now we are extracting 64 features
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    ## computes the convultional and pooling layers as above
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    ## creates a weight tensor for the first fully connected layer of given shape
    ## the output of the last pooling layer is a 7x7 pixel image and we are extracing 64 features from it
    ## for each feature we extract we are 
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    ## reshapes the output of the second pooling layer in preparation for the first fully connected layer
    ## ??why??
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    ## computes the first ouput of the first fully connected layer
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#    keep_prob = tf.placeholder(tf.float32)
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

<<<<<<< HEAD
##  This is commented out becuase we run the softmax in the training.  This is only for convenience so that you only need to look at the training function to select proper training attributes (otherwise youd need to look up here to see what the activation of each output is
=======
>>>>>>> 2599c0ed58c7b3bced08886cedeb0727e58d3f73
#    output=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    output=tf.matmul(h_fc1, W_fc2) + b_fc2
    return output

def train_neural_network(x, network_model):
    prediction = network_model(x)
<<<<<<< HEAD
    #cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#logit - unnormalized log of probability
#softmax - allows normalization of values
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#tf.equal - returns the truth value of whether the 2 arguments were equal.
#checks whether for the first dimension the largest probabiliry with one hot encoding is the same with the predicted and actual values. 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#reduce_mean reduces all dimensions by computing the mean.
=======
#    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
>>>>>>> 2599c0ed58c7b3bced08886cedeb0727e58d3f73
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

def other_train_neural_network(x, network_model):
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
<<<<<<< HEAD
#Calculates loss for the batch for reporting.
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#The following execute both of the above training methods on the fully connected neural network model.  These can also be run on the convolutional model by replacing the model name being passed to the training function
=======
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

>>>>>>> 2599c0ed58c7b3bced08886cedeb0727e58d3f73
train_neural_network(x, neural_network_model)
other_train_neural_network(x, neural_network_model)

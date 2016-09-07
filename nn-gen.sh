#!/bin/bash
#simple bash script to create a starting template for a neural network in tensorflow

#usage: expects name of python output file, number of hidden layers, followed by number of nodes in each layer 
# All hidden layers get the same number of nodes, if you would like to change that simply do so after creating the template
# start configuration of python file:

if [ $# -eq 0 ]  # Must have command-line args to demo script.
then
    echo "Please invoke this script with one or more command-line arguments (see header of script)."
    exit $E_NO_ARGS
fi

echo "import tensorflow as tf

#need to import data here
#mnist data loaded here, change!
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)
" > $1

#create number of nodes in each hidden layer
for i in $(seq 1 $2)
do
    echo "n_nodes_hl$i = $3" >> $1  
done

echo "
##mnist dependent data below (inputs and classes) change for other data sets
n_classes = 10 ###define number classes here
n_inputs = 784
batch_size = 100

x = tf.placeholder('float',[None, n_inputs])
y = tf.placeholder('float')

def neural_network_model(data):" >> $1

for i in $(seq 1 $2)
do
    if [ $i -eq 1 ]
    then
	echo "    hidden_"$i"_layer = {'weights':tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}" >> $1
    else
	echo "    hidden_"$i"_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl$((i-1)), n_nodes_hl$i])),'biases':tf.Variable(tf.random_normal([n_nodes_hl$i]))}" >> $1
    fi
done

echo "    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl$2, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
" >> $1

for i in $(seq 1 $2)
do
    if [ $i -eq 1 ]
    then
	echo "    l$i = tf.add(tf.matmul(data, hidden_"$i"_layer['weights']), hidden_"$i"_layer['biases'])
    l$i = tf.nn.relu(l$i)
" >> $1
    else
	echo "    l$i = tf.add(tf.matmul(l$((i-1)), hidden_"$i"_layer['weights']), hidden_"$i"_layer['biases'])
    l$i = tf.nn.relu(l$i)
" >> $1
    fi
done
echo "    output = tf.matmul(l$2, output_layer['weights']) + output_layer['biases']
    return output" >> $1

echo "
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	hm_epochs = 10
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			
			##mnist-data is used here, switch
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		
		##mnist data used here, change
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
" >> $1

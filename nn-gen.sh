#!/bin/bash
#simple bash script to create a starting template for a neural network in tensorflow

#usage: expects name of python output file, number of hidden layers, followed by number of nodes in each layer 
# max number of layers = 10 only for making this not generate crap tons of code
# All hidden layers get the same number of nodes, if you would like to change that simply do so after creating the template

# start configuration of python file:

if [ $# -eq 0 ]  # Must have command-line args to demo script.
then
    echo "Please invoke this script with one or more command-line arguments (see header of script)."
    exit $E_NO_ARGS
fi

echo "import tensorflow as tf

#need to import data here" >> $1

#create number of nodes in each hidden layer
for i in $(seq 1 $2)
do
    echo "n_nodes_hl$i = $3" >> $1  
done

echo "n_classes = 10 ###define number classes here
batch_size = 100
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):" >> $1

for i in $(seq 1 $2)
do
    if [ $i -eq 1 ]
    then
	echo "	hidden_"$i"_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}" >> $1
    else
	echo "	hidden_"$i"_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl$((i-1)), n_nodes_hl$i])),'biases':tf.Variable(tf.random_normal([n_nodes_hl$i]))}" >> $1
    fi
done

echo "	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl$2, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}" >> $1

for i in $(seq 1 $2)
do
    if [ $i -eq 1 ]
    then
	echo "	l$i = tf.add(tf.matmul(data, hidden_"$i"_layer['weights']), hidden_"$i"_layer['biases'])
    l$i = tf.nn.relu(l$i)" >> $1
    else
	echo "	l$i = tf.add(tf.matmul(l$((i-1)), hidden_"$i"_layer['weights']), hidden_"$i"_layer['biases'])
    l$i = tf.nn.relu(l$i)" >> $1
    fi
done
echo "	output = tf.matmul(l$2, output_layer['weights']) + output_layer['biases']
    return output" >> $1

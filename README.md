# tensor-nn-script
mnist.py is now the most up to date of files.

it takes 2 forms of input upon execution: int(number of hidden layers) 
and either a single integer value of the number of nodes to use in all hidden layers
OR a number (Y) of integers to be used as the number of nodes in each hidden layer
where Y = the number of hidden layers

examples:
python mnist.py 5 100                  ## 5 hidden layers with 100 nodes in each layer
python mnist.py 5 100 200 300 400 500  ## 5 hidden layes with each alyer having 100 more nodes than the one before it

BAD example
python 5 100 200

This repository makes it easier to follow the video here: https://www.youtube.com/watch?v=oYbVFhK_olY

In the video he references using for loops to automatically generate a python file that is ready to execute (operating on the mnist data set)

This script expects to be executed in bash with the following arguments:
$ ./nn-gen.sh filename.py num_hl num_nodes

where filename.py will be created (!! OR OVERWRITTEN!!) and num_hl and num_nodes are integer values for the number of hidden layers and the number of nodes in each hidden layer respectively.

The filename.py can then be executed without further editing:
$ python filename.py

OR can be edited for use with other data sets (look for comments with mnist

Most of this code is derived from https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
And can be found in the tensorflow github repos.

Other code came from https://www.youtube.com/user/sentdex who has some awesome videos on machine learning and tensorflow.

# tensor-nn-script
deep.py added
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

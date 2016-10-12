# tensor-nn-script
This project is a mashup of the following:
Deep MNIST for Experts: https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
and 
Other code came from https://www.youtube.com/user/sentdex who has some awesome videos on machine learning and tensorflow.

The intent was to familiarize myself with tensorflow by knocking out a quick project that was mostly based on the above examples.  The neural network models from the above examples are in each of my examples below, as well as the different training methods.  I have created two different python files for this purpose.

commented.py
This is the basic code with hard coded values for the fully connected neural network parameters.
The whole file is commented with links to the tensorflow.org site that defines each of the functions used in the program

mnist.py
This file was created to test multiple fully connect4ed neural network models with different parameters quickly.

it takes 2 forms of input upon execution: int(number of hidden layers) and either a single integer value of the number of nodes to use in all hidden layers OR a number (Y) of integers to be used as the number of nodes in each hidden layer where Y = the number of hidden layers

examples:
python mnist.py 5 100                  ## 5 hidden layers with 100 nodes in each layer
python mnist.py 5 100 200 300 400 500  ## 5 hidden layes with each alyer having 100 more nodes than the one before it

BAD example (will not run becuase not enough data to build the fully connected neural network model)
python 5 100 200

Most of this code is derived from https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
And can be found in the tensorflow github repos.

Other code came from https://www.youtube.com/user/sentdex who has some awesome videos on machine learning and tensorflow.

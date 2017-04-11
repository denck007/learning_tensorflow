# learning_tensorflow
Playing around with toy data to understand how tensorflow works

Code was written using Jupyter and Tensorflow 1.1.0-rc0 using the official docker image

This project is simply a collection of things I am working through to get a feel for how to work with Tensorflow. 

## Linear regression:
* Simple linear regression, nothing fancy. 

## Polynomial regression:
* Set up so that all controls are at the top. 
* Implemented regularization
* Generic method of dealing with powers, so additional powers only requrire the change of 1 numbner

## MNIST_FC:
* Fully connected neural network
* Set up so that there can be any number of hidden layers of any size 
* Utility functions in getMNIST.py to download and extract the images. The original code came from the tf tutorials
* Saving and loading model
* Confusion plots for training and test data

## MNIST_FCNN:
* Fully convolutional neural network
* Is intended to be generic to the number of and shapes of layers, but it really finicky
* Experimented with tensorboard. Started later in the project as a debugging activity so only prints out the net. The way it is done makes the graph really messy. Need to look into how to control the outputs better.
* I found it really interesting how small changes in the initialization caused major issues in training. I knew that it was an issue, but it was tough to track down.

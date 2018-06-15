The neural net needs to learn the following data:

X1, X2, X3, Y
[1,  1,  1,  1]
[1,  1,  0,  1],
[1,  0,  1,  0],
[1,  0,  0,  1],
[0,  1,  1,  0],
[0,  1,  0,  1],
[0,  0,  1,  1],
[0,  0,  0,  0]

Which is a single training epoch.

We try 2 network structures, one with a single hidden layer of 5 neurons, and another with 2 hidden layers of 5 neurons.
Both have 3 input neurons in the input layer and 1 in the output layer.
The weights are initially random, and the network trained by repeating epoch training data of forward calculation and back propagation for each sample, until the mean square error is < .001

The weights in the netowrk are changed by the equivalent of a stochastic gradient descent, since error corrections are made after the iteration for each sample in the epoch.
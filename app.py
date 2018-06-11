from Neuron import *
import matplotlib.pyplot as plt
import random

INPUT1 = "InputNode1"
INPUT2 = "InputNode2"
INPUT3 = "InputNode3"
OUTPUT = "Output"
NEURON_NAMES = [INPUT1, INPUT2, INPUT3, "H1_1", "H1_2", "H1_3", "H1_4", "H1_5", "H2_1", "H2_2", "H2_3", "H2_4", "H2_5", OUTPUT]


ALPHA = .1
ERROR_LIM = .001

neuron_map = {}
layers = []

epoch =     [[1, 1, 1, 1],
             [1, 1, 0, 1],
             [1, 0, 1, 0],
             [1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 1],
             [0, 0, 0, 0]]

def main():
    # set up graph
    plot_mse_figure()

    for n in range(1):
        # set up network with 1 hidden layer
        layer_node_names =  [ [NEURON_NAMES[0], NEURON_NAMES[1], NEURON_NAMES[2]],
                  [NEURON_NAMES[3], NEURON_NAMES[4], NEURON_NAMES[5], NEURON_NAMES[6], NEURON_NAMES[7]],
                  [NEURON_NAMES[13] ] ]
        # create neurons and put them in the layers list
        establish_network_structure(layer_node_names)
        # link neurons together
        initialize_connections()
        # train network by repeating iterations across epoch until mean squared error goes below limit
        # result is number of epochs and the mean_squared_error for each epoch
        epoch_count, mean_squared_error = train_network()

        labText = "1 Hidden Layer"
        # labText = "1 Hidden Layer, Trial %d"%(n+1)

        # plot network training results
        plt.plot(epoch_count, mean_squared_error, label=labText, c='blue', lw=1.5)

    for n in range(1):
        # Repeat process with network of 2 hidden layers
        layer_node_names = [[NEURON_NAMES[0], NEURON_NAMES[1], NEURON_NAMES[2]],
                            [NEURON_NAMES[3], NEURON_NAMES[4], NEURON_NAMES[5], NEURON_NAMES[6], NEURON_NAMES[7]],
                            [NEURON_NAMES[8], NEURON_NAMES[9], NEURON_NAMES[10], NEURON_NAMES[11], NEURON_NAMES[12]],
                            [NEURON_NAMES[13] ] ]

        establish_network_structure(layer_node_names)

        initialize_connections()

        epoch_count, mean_squared_error = train_network()

        labText = "2 Hidden Layers"
        #labText = "2 Hidden Layers, Trial %d"%(n+1)
        plt.plot(epoch_count, mean_squared_error, label=labText, lw=1.5, c='red')

    # display graph
    plt.legend()
    plt.show()

    return


def train_network():

    numEpochs = 0
    print("\n Initial weights and thresholds:")
    displayOutput(numEpochs, neuron_map)
    display_thresholds(layers)
    print("\n\n")

    mean_squared_error = []
    epoch_count = []

    while True:
        epoch_res = train_epoch()
        epoch_error = epoch_res['mse']

        mean_squared_error.append(epoch_error)
        epoch_count.append(numEpochs)

        numEpochs += 1

        if numEpochs % 1000 == 0:
            print("trained epochs: " + str(numEpochs))
            # print("error="+ str(epoch_error))

            displayOutput(numEpochs, neuron_map)
            display_epoch_detail(epoch_res)

        if epoch_error < ERROR_LIM:
            break

    print("------------\n\n")

    test_result(neuron_map, numEpochs, mean_squared_error[-1])

    return (epoch_count, mean_squared_error)


def plot_mse_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle("Mean Squared Error of Neural Net Epoch Training \n  Compare Number of Hidden Layers", fontsize=16)
    ax.set_xlabel('Number of Epochs', fontsize=14)
    ax.set_ylabel('Mean Squared Error', fontsize=14)
    plt.ylim(0, .25)

    ax.set_autoscaley_on(False)

    # plt.plot(num_epochs, mean_squared_error, label=labelText)
    return fig



def establish_network_structure(network_nodes):
    global layers
    layers = []
    for layer in network_nodes:
        network_layer = []
        for node_name in layer:
            neuron = Neuron(node_name)
            neuron_map[node_name] = neuron
            network_layer.append(neuron)
        layers.append(network_layer)


def initialize_connections():

    numLayers = len(layers)-1
    for i, layer in enumerate(layers[:numLayers]):
        # if i < len(layers)-1:
        for node in layer:
            for node_ahead in layers[i + 1]:
                node.add_node_ahead(node_ahead)
                node.set_weight_to_node(node_ahead, Neuron.random_number())
                print("connecting " + node.name + " forward to " + node_ahead.name)
                node_ahead.add_node_behind(node)
                print("connecting " + node_ahead.name + " backward to " + node.name)

    return layers


def train_epoch():
    epoch_res ={}
    meanSquaredError = 0

    for input_output in epoch:
        error = single_iteration(**{INPUT1: input_output[0], INPUT2: input_output[1], INPUT3: input_output[2], OUTPUT: input_output[3]})
        meanSquaredError += error**2
        res = {}
        res['Input1'] = input_output[0]
        res['Input2'] = input_output[1]
        res['Input3'] = input_output[2]
        res['Output'] = -1*(error - input_output[3])
        res['Error'] = error
        epoch_label = "epoch " + str(input_output[0]) + str(input_output[1]) + str(input_output[2]) + str(input_output[3])
        epoch_res[epoch_label] = res

    epoch_res['mse'] = meanSquaredError/8
    return epoch_res


def single_iteration(**kwargs):
    input = {k: kwargs[k] for k in (INPUT1, INPUT2, INPUT3)}
    output = kwargs.get(OUTPUT)
    # run forwards calculation to get output for each neuron
    forward_calc(**input)
     # compute delta values for each neuron
    compute_gradient(output)
    # update the weights
    update_weights_thresholds()
    # compare calculated value in output neuron with actual output
    error = output - neuron_map.get(OUTPUT).yValue
    return error


# starts forward calculation, accepts each input layer neuron name and its raw input value
def forward_calc(**input_kwargs):
    # start at input layer, move right towards output
    for layer in layers:
        for node in layer:
            calculate_output(node, **input_kwargs)


def calculate_output(node, **input_kwargs):
    input_neuron_names = (n.name for n in layers[0])
    # if the node is in the input layer, perform no calculation and pass in raw value as output
    if node in layers[0]:
        node.yValue = input_kwargs.get(node.name)
        return node.yValue

    # assume that output of neurons in previous layer have been calculated
    # calculate output of neuron based on weights and output of other neurons
    else:
        # for all nodes in previous layer (node.ahead_of_nodes), take sum of (output from nodeBehind * weight from nodeBehind to node), then subtract node.threshold
        # node.yValue = Neuron.sigmoid(sum(calculate_output(nodeBehind,x1,x2,x3) * nodeBehind.weight_to_nodes[node.name] for nodeBehind in node.ahead_of_nodes) - node.threshold)
        node.yValue = Neuron.sigmoid(sum( nodeBehind.yValue * nodeBehind.weight_to_nodes[node.name] for nodeBehind in node.ahead_of_nodes) - node.threshold)

        return node.yValue



#backwards prop - assumes a single output neuron
def compute_gradient(outVal):
    # start at output layer, calculate deltas for all nodes in layer
    # then move left towards input layer
    for layer in reversed(layers[1:]):
        for node in layer:
            calculate_delta(node, outVal)


def calculate_delta(node, outVal):
    if node.name == OUTPUT:
        node.delta = node.yValue * (1 - node.yValue) * (outVal - node.yValue)
        return node.delta
    # assume that delta for nodes in layer to the right have been calculated
    else:
        node.delta = node.yValue * (1 - node.yValue) * sum( nodeAhead.delta * node.weight_to_nodes[nodeAhead.name] for nodeAhead in node.behind_nodes)
        # print("delta=",node.delta)
        return node.delta


# call once all outputs and deltas have been calculated to update weights
def update_weights_thresholds():
    for layer in reversed(layers[1:]):
        for node in layer:
            calc_threshold_and_weight_change(node)

def calc_threshold_and_weight_change(to_node):
    threshold_change = ALPHA * -1 * to_node.delta
    to_node.threshold += threshold_change
    for node_behind in to_node.ahead_of_nodes:
        weight_change = ALPHA * node_behind.yValue * to_node.delta
        node_behind.set_weight_to_node(to_node, node_behind.get_weight_to_node(to_node) + weight_change )



def displayOutput(iteration, nm):
    print("Weights at iteration " + str(iteration))
    print(" ------ ")
    for layer in layers:
        for neuron in layer:
            neuron.display_weights()
    print("----------")

def display_deltas(layers):
    print("Delta Values")
    for layer in layers:
        for neuron in layer:
            print(neuron.name + " = " + str(neuron.delta))
    print("---")

def display_thresholds(layers):
    print("Threshold Values:")
    for layer in layers:
        for neuron in layer:
            print(neuron.name +  " = " + str(neuron.threshold) )
    print("---")

def display_epoch_detail(epoch):
    for key, trial in epoch.items():
        if key != 'mse':
            print("Inputs: " + str(trial['Input1']) + ", " + str(trial['Input2']) + ", "+ str(trial['Input3'])+ "; Output: " + str(trial['Output']) + " ; Error: " + str(trial['Error']))

    print("\n Mean Squared Error: " + str(epoch['mse']))
    print("----------\n")


def test_result(neuron_map, num, mse):

    res = {}
    # out = {}
    outputLayer = len(layers) - 1

    for sample in epoch:
        label_result = "x:"
        for input in sample[:-1]:
           label_result += '_' + str(input)
        label_result += ", y: " + str(sample[3])
        calculated = single_iteration(**{INPUT1: sample[0], INPUT2: sample[1], INPUT3: sample[2], OUTPUT: sample[3] })
        res[label_result] = (calculated, layers[outputLayer][0].yValue )

    print('Number of Epochs to solution = ' +  str(num))
    displayOutput(num, neuron_map)
    display_thresholds(layers)
    print("--Result test of trained weights--")
    for key, y_val in res.items():
        print("For inputs  " + str(key) + ", Calculated Output is:  " + str(y_val[1]))

    print("\n----------")
    print("Mean Squared Error: " + str(mse))


if __name__ == '__main__':
    main()
import numpy as np
import random
import math

#generate network
def generate(input_size, hidden_layers, output_size, weight_size, biass_size):
    network = []

    #handles input layers
    input_value = {'type': 'input', 'weight': []}
    for _ in range(input_size):
        input_value['weight'].append(random.randint(weight_size[0]*100, weight_size[1]*100)/100)

    network.append(input_value)

    #handles hidden layers
    for i in range(len(hidden_layers)):
        hidden_layer_value = {'type': 'layer', 'weight': [], 'biass': []}

        for _ in range(hidden_layers[i]):
            hidden_layer_value['weight'].append(random.randint(weight_size[0]*100, weight_size[1]*100)/100)
            hidden_layer_value['biass'].append(random.randint(biass_size[0]*100, biass_size[1]*100)/100)

        network.append(hidden_layer_value)

    #handles output layers
    output_value = {'type': 'output', 'biass': []}

    for _ in range(output_size):
        output_value['biass'].append(random.randint(biass_size[0]*100, biass_size[1]*100)/100)

    network.append(output_value)

    return network

#activation functions
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

#calculate network
def calculate(network, inputs, activation_name):
    for place in range(len(network)):
        if network[place]['type'] == 'input':
            network[place]['values'] = inputs
        else:
            values = []
            
            for biass in network[place]['biass']:
                value = np.multiply(network[place - 1]['values'], network[place - 1]['weight'])
                value = np.append(value, [biass])

                if activation_name == 'sigmoid':
                    value = sigmoid(np.sum(value))

                values.append(value)
            
            network[place]['values'] = values

    return network[len(network)-1]['values']

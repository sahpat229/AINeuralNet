import numpy as np 

def parse_representation(representation_file_name):
    repr_file = open(representation_file_name, 'r')
    input_nodes, hidden_nodes, output_nodes = map(int, next(repr_file).split())

    hidden_weights = np.zeros((input_nodes+1, hidden_nodes))
    output_weights = np.zeros((hidden_nodes+1, output_nodes))

    index = 0
    for line in repr_file:
        weights = list(map(float, line.split()))
        if index < hidden_nodes:
            hidden_weights[:, index] = weights
        else:
            output_weights[:, index-hidden_nodes] = weights
        index += 1

    return [input_nodes, hidden_nodes, output_nodes], [hidden_weights, output_weights]

def parse_training(training_file_name):
    train_file = open(training_file_name, 'r')
    num_examples, input_nodes, output_nodes = map(int, next(train_file).split())

    inputs = np.zeros((num_examples, input_nodes))
    labels = np.zeros((num_examples, output_nodes))

    index = 0
    for line in train_file:
        variables = list(map(float, line.split()))
        inputs[index, :] = variables[:input_nodes]
        labels[index, :] = variables[input_nodes:]
        index += 1

    return inputs, labels

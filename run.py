# Sahil Patel
# AI Neural Network Project

import argparse
import input_lib
import network_lib

parser = argparse.ArgumentParser(description="Runnable file for AI Neural Network Project")
parser.add_argument('--type', action='store', type=str)
namespace = parser.parse_args()

representation_file_name = input("Enter the name of the neural network's representation file: ")
nodes, weights = input_lib.parse_representation(representation_file_name=representation_file_name)
network = network_lib.Network(nodes=nodes,
                              weights=weights)
if namespace.type == 'train':
    training_file_name = input("Enter the name of the training file: ")
    inputs, labels = input_lib.parse_training(training_file_name=training_file_name)
    output_file_name = input("Enter the name of the output file: ")
    epochs = int(input("Enter the number of epochs: "))
    learning_rate = float(input("Enter the learning rate: "))
    network.train(inputs=inputs,
                  labels=labels,
                  learning_rate=learning_rate,
                  epochs=epochs)
    network.output_trained(output_file_name)
else:
    testing_file_name = input("Enter the name of the testing file: ")
    inputs, labels = input_lib.parse_training(training_file_name=testing_file_name)
    output_file_name = input("Enter the name of the output file: ")
    network.test(inputs=inputs,
                 labels=labels)
    network.output_test_results(output_file_name)

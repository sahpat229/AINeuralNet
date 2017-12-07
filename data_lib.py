import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_data(means, covs, pis, num_samples):
    # N clusters
    # means : [N x 2]
    # covs : [N x 2 x 2]
    # pis : [N]

    data = []
    cluster_labels = []
    num_clusters = means.shape[0]
    for sample in range(num_samples):
        indices = list(range(num_clusters))
        index = int(np.random.choice(indices, p=pis))
        data.append(np.random.multivariate_normal(mean=means[index],
                                                  cov=covs[index]))
        cluster_labels.append(index)
    data = np.array(data)
    cluster_labels = np.array(cluster_labels)

    # change labels to be one hot vectors
    labels = np.zeros((num_samples, num_clusters))
    labels[np.arange(num_samples), cluster_labels] = 1

    return data, labels

def normalize_data(data, scaler=None):
    if scaler is not None:
        data = scaler.transform(data)
        return data, scaler

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler

def create_weights(input_nodes, hidden_nodes, output_nodes, variance):
    hidden_weights = np.random.randn(input_nodes+1, hidden_nodes)*np.sqrt(variance)
    output_weights = np.random.randn(hidden_nodes+1, output_nodes)*np.sqrt(variance)
    return hidden_weights, output_weights

def output_training_data(data, labels, input_nodes, output_nodes, output_file_name):
    output_file = open(output_file_name, 'w+')

    num_examples, input_nodes, output_nodes = (data.shape[0], data.shape[1], labels.shape[1])
    output_file.write("{:d} {:d} {:d}\n".format(int(num_examples),
                                                int(input_nodes),
                                                int(output_nodes)))
    for index, (example, label) in enumerate(zip(data, labels)):
        output_file.write(" ".join(["%.3f" % example_item for example_item in example.tolist()]))
        output_file.write(" ")
        output_file.write(" ".join(["%d" % label_item for label_item in label.tolist()]))
        if index != data.shape[0] - 1:
            output_file.write("\n")

    output_file.close()

def output_representation(input_nodes, hidden_nodes, output_nodes, hidden_weights, output_weights,
                          output_file_name):
    assert input_nodes == hidden_weights.shape[0] - 1
    assert hidden_nodes == hidden_weights.shape[1]
    assert hidden_nodes == output_weights.shape[0] - 1
    assert output_nodes == output_weights.shape[1]

    output_file = open(output_file_name, 'w+')
    output_file.write("{:d} {:d} {:d}\n".format(int(input_nodes),
                                                int(hidden_nodes),
                                                int(output_nodes)))
    for col in range(hidden_weights.shape[1]):
        line = " ".join(["%.3f" % weight for weight in hidden_weights[:, col]])
        output_file.write(line + "\n")

    for col in range(output_weights.shape[1]):
        line = " ".join(["%.3f" % weight for weight in output_weights[:, col]])
        output_file.write(line)
        if col != output_weights.shape[1] - 1:
            output_file.write("\n")

    output_file.close()

class CustomFiles():
    def __init__(self, means, covs, pis, num_train_samples, num_test_samples,
                 output_train_file_name, output_test_file_name,
                 input_nodes, hidden_nodes, output_nodes, variance,
                 output_repr_file_name):
        train_data, train_labels = create_data(means=means,
                                               covs=covs,
                                               pis=pis,
                                               num_samples=num_train_samples)
        test_data, test_labels = create_data(means=means,
                                             covs=covs,
                                             pis=pis,
                                             num_samples=num_test_samples)
        train_data, scaler = normalize_data(data=train_data,
                                            scaler=None)
        test_data, scaler = normalize_data(data=test_data,
                                           scaler=scaler)
        output_training_data(data=train_data,
                             labels=train_labels,
                             input_nodes=input_nodes,
                             output_nodes=output_nodes,
                             output_file_name=output_train_file_name)
        output_training_data(data=test_data,
                             labels=test_labels,
                             input_nodes=input_nodes,
                             output_nodes=output_nodes,
                             output_file_name=output_test_file_name)
        hidden_weights, output_weights = create_weights(input_nodes=input_nodes,
                                                        hidden_nodes=hidden_nodes,
                                                        output_nodes=output_nodes,
                                                        variance=variance)
        output_representation(input_nodes=input_nodes,
                              hidden_nodes=hidden_nodes,
                              output_nodes=output_nodes,
                              hidden_weights=hidden_weights,
                              output_weights=output_weights,
                              output_file_name=output_repr_file_name)
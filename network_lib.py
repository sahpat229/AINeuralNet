import numpy as np 
from scipy.special import expit

class Network():
    def __init__(self, nodes, weights):
        self.nodes = nodes
        self.weights = weights

    def forward_prop(self, inp):
        for weight_matrix in self.weights:
            inp = np.concatenate(([-1], inp), axis=0)
            inp = np.transpose(weight_matrix).dot(inp)
            inp = expit(inp)
        return inp

    def train_step(self, inp, lab, learning_rate):
        activations = []
        for weight_matrix in self.weights:
            inp = np.concatenate(([-1], inp), axis=0)
            activations.append(inp)
            inp = np.transpose(weight_matrix).dot(inp)
            inp = expit(inp)
        activations.append(inp)

        delta_last = (lab - activations[2]) * activations[2] * (1 - activations[2])
        delta_middle = (self.weights[1].dot(delta_last)) * activations[1] * (1 - activations[1])
        [delta_last, delta_middle] = [np.expand_dims(delta, axis=0) for delta in [delta_last, delta_middle[1:]]]
        activations = [np.expand_dims(activation, axis=1) for activation in activations]

        self.weights[0] += learning_rate * activations[0].dot(delta_middle)
        self.weights[1] += learning_rate * activations[1].dot(delta_last)

    def train(self, inputs, labels, learning_rate, epochs):
        for _ in range(epochs):
            for inp, lab in zip(inputs, labels):
                self.train_step(inp, lab, learning_rate)

    def test_step(self, inp, lab):
        sigm_outputs = self.forward_prop(inp)
        class_outputs = (sigm_outputs >= 0.5).astype(np.int)
        lab = lab.astype(np.int)
        for index, class_output in enumerate(class_outputs):
            self.confusion_matrices[index][class_output][lab[index]] += 1

    def compute_individual_metrics(self, confusion_matrix):
        accuracy = (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
        precision = confusion_matrix[1, 1] / np.sum(confusion_matrix[1, :])
        recall = confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
        f1 = (2 * precision * recall) / (precision + recall)
        return [accuracy, precision, recall, f1]

    def compute_overall_metrics(self):
        overall_table = np.sum(self.confusion_matrices, axis=0)
        micro_averaged = self.compute_individual_metrics(overall_table)

        individual_results = np.zeros((len(self.confusion_matrices), 4))
        for ind in range(len(self.confusion_matrices)):
            individual_results[ind, :] = self.compute_individual_metrics(self.confusion_matrices[ind])

        macro_averaged = np.mean(individual_results, axis=0)
        macro_f1 = (2 * macro_averaged[1] * macro_averaged[2]) / (macro_averaged[1] + macro_averaged[2])
        macro_averaged = macro_averaged[0:3].tolist() + [macro_f1]

        return individual_results, micro_averaged, macro_averaged

    def test(self, inputs, labels):
        self.confusion_matrices = [np.zeros((2, 2), dtype=np.int) for _ in range(labels.shape[1])]
        for inp, lab in zip(inputs, labels):
            self.test_step(inp, lab)

    def output_trained(self, file_name):
        output_file = open(file_name, 'w+')
        output_file.write(" ".join(map(str, self.nodes)) + "\n")

        for col in range(self.weights[0].shape[1]):
            line = " ".join(["%.3f" % weight for weight in self.weights[0][:, col]])
            output_file.write(line + "\n")

        for col in range(self.weights[1].shape[1]):
            line = " ".join(["%.3f" % weight for weight in self.weights[1][:, col]])
            output_file.write(line)
            if col != self.weights[1].shape[1] - 1:
                output_file.write("\n")

        output_file.close()

    def output_test_results(self, file_name):
        individual_results, micro_averaged, macro_averaged = self.compute_overall_metrics()
        individual_confusions = [[confusion_matrix[1, 1], confusion_matrix[1, 0], 
                                  confusion_matrix[0, 1], confusion_matrix[0, 0]]
                                  for confusion_matrix in self.confusion_matrices]

        output_file = open(file_name, 'w+')
        for individual_confusion, individual_result in zip(individual_confusions, individual_results):
            output_file.write(" ".join(map(str, individual_confusion)) + " ")
            output_file.write(" ".join(["%.3f" % metric for metric in individual_result.tolist()]) + "\n")

        output_file.write(" ".join(["%.3f" % metric for metric in micro_averaged]) + "\n")
        output_file.write(" ".join(["%.3f" % metric for metric in macro_averaged]))

        output_file.close()
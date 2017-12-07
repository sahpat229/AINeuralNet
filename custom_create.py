#Sahil Patel
#AI Neural Network Project

from data_lib import *
import os

means = np.array([[3, 3], [0.5, -1.5], [1.5, -0.5], [-3, -3]])
covs = np.array([[[1, 0],
                  [0, 1]],
                 [[1, 0.5],
                  [0.5, 1]],
                 [[1, 0.2],
                  [0.2, 1]],
                 [[1, 0.1],
                  [0.1, 1]]])
print(covs[0].shape)
pis = np.array([0.3, 0.2, 0.2, 0.3])
num_train_samples = 1000
num_test_samples = 100

output_path = './CustomTextFiles/'

output_train_file_name = os.path.join(output_path, 'train.txt')
output_test_file_name = os.path.join(output_path, 'test.txt')
output_repr_file_name = os.path.join(output_path, 'repr.txt')

input_nodes = means.shape[1]
output_nodes = means.shape[0]
hidden_nodes = 30
variance = 0.5

CustomFiles(means=means, 
            covs=covs, 
            pis=pis, 
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            output_train_file_name=output_train_file_name,
            output_test_file_name=output_test_file_name,
            input_nodes=input_nodes,
            hidden_nodes=hidden_nodes,
            output_nodes=output_nodes,
            variance=variance,
            output_repr_file_name=output_repr_file_name)
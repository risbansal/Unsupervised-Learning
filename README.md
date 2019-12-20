# Unsupervised-Learning
Unsupervised learning using Bayesian networks and Chow Liu Tree ALgorithm


Language
Python 2.7

Required Libraries:
numpy as np
time
scipy.sparse import csr_matrix
scipy.sparse.csgraph import minimum_spanning_tree
collections import defaultdict
operator
warnings
argparse



Commandline options:
data_path - path of the folder containing all datasets
dataset_name - Name of dataset
algorithm - algorithm to run

1 - Independent Bayesian Network
2 - Chow Liu Tree
3 - Mixture of Chow Liu trees
4 - Chow Liu Trees with Random Forest

Example:
python code.py -data_path "hw4-datasets/small-10-datasets" -dataset_name 'accidents' -algorithm 1


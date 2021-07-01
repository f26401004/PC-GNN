from scipy.io import loadmat
import scipy.sparse as sp
from collections import defaultdict
import numpy as np

def sparse_to_adjacent_list(sparse_matrix):
  # Add self loop into the sparse matrix
  homo_adjacent_list = sparse_matrix + sp.eye(sparse_matrix.shape[0])
  adjacent_lists = defaultdict(set)
  edges = homo_adjacent_list.nonzero()
  for index, node in enumerate(edges[0]):
    adjacent_lists[node].add(edges[1][index])
    adjacent_lists[edges[1][index]].add(node)
        
  return adjacent_lists

def adjacent_list_to_sparse(adjacent_list):
  matrix = []
  for i in range(len(adjacent_list)): 
    matrix.append([0] * len(adjacent_list))
    # Map the edge
    for j in adjacent_list[i]:
      matrix[i][j] = 1
  return matrix

def load_data():
  amazon_data = loadmat('./input/Amazon.mat')
  network_upu_data = amazon_data['net_upu'] # users reviewing at least one same product
  network_usu_data = amazon_data['net_usu'] # users having at least one same star rating within one week
  network_uvu_data = amazon_data['net_uvu'] # users with top 5% mutual review text similarities
  network_homo_data = amazon_data['homo']
    
  features_data = amazon_data['features'].todense().A
  labels = amazon_data['label'].flatten()
    
  return [network_homo_data, network_upu_data, network_usu_data, network_uvu_data], features_data, labels

def positive_negative_split(nodes, labels):
  positive_nodes = []
  negative_nodes = cp.deepcopy(nodes)
  auxious_nodes = cp.deepcopy(nodes)
  for idx, label in enumerate(labels):
    if label == 1:
      positive_nodes.append(auxious_nodes[idx])
      negative_nodes.remove(auxious_nodes[idx])
    
  return positive_nodes, negative_nodes

def row_normalize(matrix):
  row_sum = np.array(matrix.sum(axis=1)) + 0.01
  row_inverse = np.power(row_sum, -1).flatten()
  row_inverse[np.isinf(row_inverse)].flatten()
  row_matrix_inverse = sp.diags(row_inverse)
  matrix = row_matrix_inverse.dot(matrix)
    
  return matrix

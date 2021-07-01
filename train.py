from utils import load_data, row_normalize, sparse_to_adjacent_list
from layers import InterAggregator
from sklearn.model_selection import train_test_split
import torch
import networkx as nx
import numpy as np
import torch.nn as nn

# Define the hyperparameters
embedding_size = 64
step_size = 2e-2
number_epochs = 100
learning_rate = 1e-1
weight_decay = 1e-3
loss_lambda = 2

# Load graph, features, and label
[network_homo_data, network_upu_matrix, network_usu_matrix, network_uvu_matrix], features_data, labels = load_data()

network_upu_adjacent = sparse_to_adjacent_list(network_upu_matrix)
network_usu_adjacent = sparse_to_adjacent_list(network_usu_matrix)
network_uvu_adjacent = sparse_to_adjacent_list(network_uvu_matrix)


# Remove 0-3304 nodes (unlabeled)
index = list(range(3305, len(labels)))
# Split the train and test data
idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                      test_size=0.60, random_state=2, shuffle=True)

# Initialize the model input
features_embedding = nn.Embedding(features_data.shape[0], features_data.shape[1])
features_data = row_normalize(features_data)
features_embedding.weight = nn.Parameter(torch.LongTensor(features_data), requires_grad=False)

relation_adjacent_lists = [network_upu_adjacent, network_usu_adjacent, network_uvu_adjacent]


# Calculate for A head matrix
all_relation_matrix = []
d_matrix = [0] * len(network_upu_adjacent)
d_matrix_pre = []
d_matrix_post = []
d_matrix_node_set = []
for i in range(len(network_upu_adjacent)):
  all_relation_matrix.append([0] * len(network_upu_adjacent))
  d_matrix_pre.append([0] * len(network_upu_adjacent))
  d_matrix_post.append([0] * len(network_upu_adjacent))
  d_matrix_node_set.append(set())
    
all_relation_matrix = np.array(all_relation_matrix)

for node in network_upu_adjacent:
  neighbor_nodes = network_upu_adjacent[node]
  if len(neighbor_nodes) == 0:
    continue
  d_matrix_node_set[node] = set.union(d_matrix_node_set[node], set(neighbor_nodes))
  for neighbor_node in neighbor_nodes:
    all_relation_matrix[node][neighbor_node] += 1
    all_relation_matrix[neighbor_node][node] += 1
        
for node in network_usu_adjacent:
  neighbor_nodes = network_usu_adjacent[node]
  if len(neighbor_nodes) == 0:
    continue
  d_matrix_node_set[node] = set.union(d_matrix_node_set[node], set(neighbor_nodes))
  for neighbor_node in neighbor_nodes:
    all_relation_matrix[node][neighbor_node] += 1
    all_relation_matrix[neighbor_node][node] += 1
        
for node in network_uvu_adjacent:
  neighbor_nodes = network_uvu_adjacent[node]
  if len(neighbor_nodes) == 0:
    continue
  d_matrix_node_set[node] = set.union(d_matrix_node_set[node], set(neighbor_nodes))
  for neighbor_node in neighbor_nodes:
    all_relation_matrix[node][neighbor_node] += 1
    all_relation_matrix[neighbor_node][node] += 1


for node in range(len(d_matrix_node_set)):
  value = len(d_matrix_node_set[node])
  d_matrix[node] = value

# Compute d**-1/2 and d**1/2
for node in range(len(d_matrix)):
  d_matrix_pre[node][node] = d_matrix[node]**(-0.5)
  d_matrix_post[node][node] = d_matrix[node]**(0.5)

all_relation_matrix_head = np.matmul(np.matmul(d_matrix_pre, all_relation_matrix), d_matrix_post)
all_relation_matrix_head = row_normalize(all_relation_matrix_head)


# Construct the graph
network_upu = nx.Graph()
for edge in network_upu_data:
    nonzeros = edge.nonzero()
    if (len(nonzeros[0]) == 0):
        continue
    node_a = nonzeros[0][0]
    for node_b in nonzeros[1]:
        network_upu.add_edge(node_a, node_b)

network_usu = nx.Graph()
for edge in network_usu_data:
    nonzeros = edge.nonzero()
    if (len(nonzeros[0]) == 0):
        continue
    node_a = nonzeros[0][0]
    for node_b in nonzeros[1]:
        network_usu.add_edge(node_a, node_b)

network_uvu = nx.Graph()
for edge in network_uvu_data:
    nonzeros = edge.nonzero()
    if (len(nonzeros[0]) == 0):
        continue
    node_a = nonzeros[0][0]
    for node_b in nonzeros[1]:
        network_uvu.add_edge(node_a, node_b)

networks = [network_upu, network_usu, network_uvu]


inter_aggregator = InterAggregator(features_data, y_train, features_data.shape[1], embedding_size,
                                   relation_adjacent_lists, networks, all_relation_matrix,
                                   step_size=step_size, cuda=True)
gnn_model = PCGNN(2, inter_aggregator, loss_lambda)

# Create Adam optimizer 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=learning_rate, weight_decay=weight_decay)

# Get the the number of label
number_train_positive = len(list(filter(lambda x: x == 1, y_train)))
number_train_negative = len(y_train) - number_train_positive

for epoch in range(number_epoch):
  start_time = time.time()

  epoch_picked_nodes = []
  epoch_picked_labels = []
  epoch_picked_features = []
    
  for index in range(len(networks)):
    # Pick the nodes with comupted probability
    picked_nodes = []
            
    for index, node in enumerate(idx_train):
      # Get label ferquency of the node
      node_label = y_train[index]
      label_frequency = (number_train_positive if node_label == 1 else number_train_negative) / len(idx_train)
      # Get the A head
      p = (all_relation_matrix_head[:, node].sum())**2 / label_frequency
      r = random.uniform(0, 1)
      if (r <= p):
        picked_nodes.append(node)
        
    epoch_picked_nodes.append(picked_nodes)
    epoch_picked_labels.append(labels[np.array(picked_nodes)])
    epoch_picked_features.append(features_data[np.array(picked_nodes)])


  loss = gnn_model.loss(epoch_picked_nodes, epoch_picked_labels, epoch_picked_features)
  loss.backward()
  optimizer.step()
  end_time = time.time()
  epoch_time += end_time - start_time
  loss += loss.item()

  print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
    
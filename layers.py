import torch
import torch.nn as nn
from torch.nn import init

class InterAggregator(nn.Module):
  def __init__(self, features, labels, features_dimension, embedding_dimension,
               relation_adjacent_lists, networks, all_relation_matrix,
               step_size=0.02, cuda=True):
    super(InterAggregator, self).__init__()
        
    self.features = features
    self.labels = labels
    self.features_dimension = features_dimension
    self.embedding_dimension = embedding_dimension
    self.relation_adjacent_lists = relation_adjacent_lists
    self.networks = networks
    self.all_relation_matrix = all_relation_matrix
    self.step_size = step_size
    self.cuda = cuda
        
    self.weight = nn.Parameter(torch.cuda.FloatTensor(embedding_dimension, features_dimension)) # The weight parameters to transform node embeddings before inter-aggregation
    self.weight.cuda()
    init.xavier_uniform_(self.weight)
        
    self.distance_nets = []
    for index in range(len(networks)):
      self.distance_nets.append(nn.Sequential(
        torch.nn.Linear(features_dimension, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
        torch.nn.Sigmoid()
      ))
    self.distance_nets[index].cuda('cuda:0')


  def forward(self, picked_relation_nodes, picked_relation_labels, picked_relation_features, train_flag=True):
    relation_features = []
    center_features = []
        
    # Get same label nodes
    negative_label_nodes = list(filter(lambda x: x[1] == 0, enumerate(self.labels)))
    negative_label_nodes = list(map(lambda x: x[0], negative_label_nodes))
    positive_label_nodes = list(filter(lambda x: x[1] == 1, enumerate(self.labels)))
    positive_label_nodes = list(map(lambda x: x[0], positive_label_nodes))
        
    # Extract 1-hop neighbor ids from adjacent lists of each single-relation graph
    to_neighs = []
    for index, picked_relation_node in enumerate(picked_relation_nodes):
      to_neighs.append(dict())
      adjacent_list = self.relation_adjacent_lists[index]
      for node in picked_relation_node:
        if node not in to_neighs[index]:
          to_neighs[index][node] = set()
        to_neighs[index][node] = set.union(to_neighs[index][node], set(adjacent_list[node]))
        
    # Loop for each relation graph
    for index, picked_relation_node in enumerate(picked_relation_nodes):
      # Find minority class
      number_negative = len(list(filter(lambda x: x == 0, picked_relation_labels[index])))
      number_positive = len(list(filter(lambda x: x == 1, picked_relation_labels[index])))
      if number_negative < number_positive:
        minority_class = 0
      else:
        minority_class = 1
            
      # Compute the rho plus for current relation graph
      rho_plus = 0
            
      # Get all minority class nodes
      minority_nodes = list(filter(lambda x: picked_relation_labels[index][x[0]] == minority_class, enumerate(picked_relation_node)))
      minority_nodes_index = list(map(lambda x: int(x[0]), minority_nodes))
      minority_nodes = list(map(lambda x: x[1], minority_nodes))
      average_neighborhood_size = 0
      # Get the node distance from distance net
      minority_nodes_diff_distances = []
      minority_nodes_distances = self.distance_nets[index](torch.cuda.FloatTensor(picked_relation_features[index][np.array(minority_nodes_index)]))
            
      # Compute the diff distance from each pair of minority nodes' 
      for node_a_index in tqdm(range(len(minority_nodes))):
        node_a = minority_nodes[node_a_index]
        if self.networks[index].has_node(node_a) == False:
          continue
        # Sum up the number of neighborhood nodes on each single network
        average_neighborhood_size += self.networks[index].degree(node_a)
        for node_b_index in range(node_a_index + 1, len(minority_nodes)):
          node_b = minority_nodes[node_b_index]
                    
          d_a = minority_nodes_distances[node_a_index][1]
          d_b = minority_nodes_distances[node_b_index][1]
                    
          minority_nodes_diff_distances.append(abs(d_a - d_b).item())

      average_neighborhood_size = math.floor(average_neighborhood_size / len(minority_nodes))
      minority_nodes_diff_distances = np.sort(np.array(minority_nodes_diff_distances))[::-1]
      if average_neighborhood_size != 0:
        rho_plus = np.sum(minority_nodes_diff_distances[:average_neighborhood_size]) / average_neighborhood_size
      else:
        rho_plus = 0
      print('rho_plus', rho_plus)
            
            
      # Compute the rho minus
      rho_minus = 0
      # Get the node distance from distance net
      nodes_distances = self.distance_nets[index](torch.cuda.FloatTensor(picked_relation_features[index]))
      nodes_diff_distances = []
            
            
      for node_a_index in tqdm(range(len(picked_relation_node))):
        node_a = picked_relation_node[node_a_index]
        for node_b in list(to_neighs[index][node_a]):              
          d_a = nodes_distances[node_a_index][1]
          d_b = self.distance_nets[index](torch.cuda.FloatTensor(self.features[node_b]))[1]
                    
          nodes_diff_distances.append(abs(d_a - d_b).item())
            
      nodes_diff_distances = np.sort(np.array(nodes_diff_distances))[::-1]
      rho_minus = np.sum(nodes_diff_distances[:math.floor(len(nodes_distances)/2)]) / math.floor(len(nodes_diff_distances)/2)
      print('rho_minus', rho_minus)
            
      # Get embedding features from relation graph
      relation_features.append([])
      # Under-sample the neighborhood nodes according to rho_minus
      for node_v_index in tqdm(range(len(picked_relation_node))):
        node_v = picked_relation_node[node_v_index]
        # Get the connected node from all_relation_matrix
        connected_nodes = list(filter(lambda x: x[1] != 0, enumerate(self.all_relation_matrix[node_v])))
        connected_nodes = list(map(lambda x: x[0], connected_nodes))
        for node_u in connected_nodes:
          distance_v = nodes_distances[node_v_index][1]
          distance_u = self.distance_nets[index](torch.cuda.FloatTensor([node_u]))[1]
          distance_diff = abs(distance_v - distance_u).item()
                    
          # Under-sample condition
          if distance_diff < rho_minus:
            relation_features[index].append(self.features[node_u])
            

      # Over-sample the neighborhood nodes according to rho_plus
      for node_v_index in tqdm(range(len(picked_relation_node))):
        node_v = picked_relation_node[node_v_index]
        for node_u in (negative_label_nodes if self.labels[node_v_index] == 0 else positive_label_nodes):
          distance_v = nodes_distances[node_v_index][1]
          distance_u = self.distance_nets[index](torch.cuda.FloatTensor([node_u]))[1]
          distance_diff = abs(distance_v - distance_u).item()
                    
          # Over-sampl condition
          if distance_diff < rho_plus:
            relation_features[index].append(self.features[node_u])
                        
    center_features = picked_relation_features
    neighbor_features = torch.cat(relation_features, dim=0)
        
    center_h = self.weight.mm(center_features.t())
    neigh_h = self.weight.mm(neighbor_features.t())
        
    aggregated = torch.zeros(size=(self.embedding_dimmension, n))
    for r in range(len(self.picked_relation_nodes)):
      aggregated += neigh_h[:, r * n:(r + 1) * n]
       
    combined = F.relu(center_h + aggregated)
        
    # Train the distance network (FC neural network)
    for index, distance_net in enumerate(self.distance_nets):
      loss_fnunction = nn.CrossEntropyLoss()
      y_pred = distance_net(self.features[np.array(picked_relation_nodes[index])])[:, 1]
      loss = loss_function(y_pred, labels[np.array(picked_relation_nodes[index])])
      intra_network.zero_grad()
      loss.backward()
      with torch.no_grad():
        for param in model.parameters():
          param -= 1e-4 * param.grad

    return combined
        
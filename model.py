import torch
import torch.nn as nn
from torch.nn import init

class PCGNN(nn.Module):
  def __init__(self, number_classes, inter_aggregator, loss_lambda):
    super(PCGNN, self).__init__()
    self.inter_aggregator = inter_aggregator
    self.cross_entropy = nn.CrossEntropyLoss()
    self.loss_lambda = loss_lambda
    self.weight = nn.Parameter(torch.cuda.FloatTensor(number_classes, inter_aggregator.embedding_dimension))
    init.xavier_uniform_(self.weight)
        
  def forward(self, relation_nodes, relation_labels, train_flag=True):
    embeddings, label_scores = self.inter_aggregator(relation_nodes, relation_labels, train_flag)
    scores = self.weight.mm(embeddings)
    return scores.t(), label_scores
    
  def loss(self, relation_nodes, relation_labels, train_flag=True):
    gnn_scores, label_scores = self.forward(relation_nodes, relation_labels, train_flag)
    label_loss = self.cross_entropy(label_scores, labels.squeeze())
    gnn_loss = self.cross_entropy(gnn_scores, labels.squeeze())
    final_loss = gnn_loss + self.loss_lambda * label_loss
        
    return final_loss
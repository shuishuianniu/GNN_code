import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from layer.GCN_layer import GCN_layer


class GCN(nn.Module):
    def __init__(self,in_feature,hidden,out_feature,dropout,bias=True ):
        super(GCN,self).__init__()
        self.gcn1 = GCN_layer(in_feature,hidden,bias)
        self.gcn2 = GCN_layer(hidden,out_feature,bias)
        self.dropout = dropout

    def forward(self, input, adj):
        middle = self.gcn1(input,adj)
        middle = F.relu(F.dropout(middle,self.dropout,training=True))
        out = self.gcn2(middle,adj)
        out = F.log_softmax(out, dim=1)
        return out





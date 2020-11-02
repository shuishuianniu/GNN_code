import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class GraphConv(Module):
    def __init__(self,in_features,out_features,dropout,fdim=False,bias = True):
        super(GraphConv,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight1 = Parameter(torch.FloatTensor(in_features,in_features))
        self.weight2 = Parameter(torch.FloatTensor(in_features,out_features))

        if fdim is not False:
            print("have fdim")
            self.fdim = True
            self.mlp = nn.Linear(fdim,in_features)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()


    def reset_parameters(self):
        stdv1 = 1.0 / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1,stdv1)
        stdv2 = 1.0 / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv2,stdv2)

    def forward(self,input):
        if self.fdim:
            input = self.mlp(input)
        N = torch.mm(torch.mm(input,self.weight1),input.T)
        N = F.dropout(N,self.dropout,training=self.training)
        middle = torch.mm(input,self.weight2)
        out = torch.mm(N,middle)
        if self.bias is not None:
            out = out + self.bias
        out = F.log_softmax(out, dim=1)
        return out

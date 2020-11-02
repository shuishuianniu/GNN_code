import torch.nn as nn
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class MLP(Module):
    def __init__(self,in_feature,hidden,out_feature,bias=True):
        super(MLP,self).__init__()

        self.l1 = nn.Linear(in_feature,hidden,bias=bias)
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(hidden,out_feature,bias=bias)


    def forward(self, input):
        middle = self.tanh(self.l1(input))
        middle = self.l2(middle)
        #print(middle)
        out = F.log_softmax(middle, dim=1)
        return out








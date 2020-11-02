import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math


class GCN_layer(Module):
    def __init__(self,in_feature,out_feature,bias = True):
        super(GCN_layer,self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



















#
# msg = fn.copy_src(src='h',out='m')
#
# def reduce(nodes):  #accumulation
#     accum = torch.mean(nodes.mailbox['m'],1)
#     return {'h':accum}
#
# class NodeApplyModule(nn.Module):
#     def __init__(self,in_dim,out_dim):
#         self.linear = nn.Linear(in_dim,out_dim)
#
#     def forward(self,nodes,activation):
#         h = self.linear(nodes.data['h'])
#         return {'h':h}
#
# class GCNLayer(nn.module):
#     def __init__(self,in_dim,out_dim,activation,dropout,batch_norm,residual=False,dgl_builtin=False):
#         super.__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout)
#         self.batch_norm = batch_norm
#         self.residual = residual
#         self.dgl_builtin = dgl_builtin
#
#         if in_dim != out_dim:
#             self.residual = False
#
#         if self.batch_norm:
#             self.batchnorm_h = nn.BatchNorm1d(out_dim)
#
#         if self.dgl_builtin == False:
#             self.apply_mod = NodeApplyModule(in_dim,out_dim)
#         elif dgl.__version__ < "0.5":
#             self.conv = GraphConv(in_dim,out_dim)
#         else:
#             self.conv = GraphConv(in_dim,out_dim,allow_zero_in_degree = True)
#
#
#     def forward(self,g,feature):
#         h_in = feature
#         if self.dgl_builtin == False:
#             g.ndata['h'] = feature
#             g.update_all(msg,reduce)
#             g.apply_nodes(func = self.apply_mod)
#             h = g.ndata['h']
#         else:
#             h = self.conv(g,feature)
#
#         if self.batch_norm:
#             h = self.batchnorm_h(h)
#
#         h = self.activation(h)
#
#         if self.residual:
#             h = h + h_in
#
#         h = self.dropout(h)
#
#         return h






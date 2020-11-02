import numpy as np
import torch
from scipy import sparse as sp
from layer.GraphConv import GraphConv
from layer.MLP import MLP
from net.GCN_net import GCN


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename+'.npz',allow_pickle=True)
    return sp.csr.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def gnn_model(MODEL_NAME,params,setting):
    models = {
        'topoconv':GraphConv(params['in_feature'],params['class_num'],setting['dropout'],params['fdim'],params['bias']),
        'mlp':MLP(params["fdim"],params['hidden'],params['class_num'],params['bias']),
        'gcn':GCN(params['fdim'],params['hidden'],params['class_num'],setting['dropout'],params['bias'])
    }
    return models[MODEL_NAME],MODEL_NAME
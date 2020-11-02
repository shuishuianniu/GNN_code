import os
import dgl
import numpy as np
from scipy import sparse as sp
from utils.utils import  save_sparse_csr
from utils.utils import load_sparse_csr
from utils.utils import sparse_mx_to_torch_sparse_tensor
import torch


def prepare(params,args):
    edges = np.genfromtxt(params['edge_path'],dtype=np.int32)
    if params['has_fea']:
        features = np.genfromtxt(params['feature_path'],dtype=np.float)
    else:
        features = []
    labels = np.loadtxt(params['label_path'],dtype=np.int32)

    src,dst = np.split(edges,edges.shape[1],axis=1)
    nodes = labels.size


    g = dgl.DGLGraph()
    g.add_nodes(nodes)
    if params['has_fea']:
        g.ndata['feat'] = features
    g.add_edges(list(src.reshape(src.size)),list(dst.reshape(dst.size)))
    g.add_edges(list(dst.reshape(dst.size)),list(src.reshape(src.size)))
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
    NA = N * A

    if os.path.exists(params['NA3_path']):
        NA3=load_sparse_csr(params["NA3_path"])
    else:
        NA3=NA*NA*NA
        save_sparse_csr(params["NA3_path"], NA3)

    if os.path.exists(params['topofea_path'].format(params['in_feature'])):
        topofea = np.loadtxt(params["topofea_path"].format(params['in_feature']), dtype=float)

    else:
        print("not exist topofea")
        (col, row) = (NA3 != 0).nonzero()
        col = col.tolist()
        row = row.tolist()
        topo = [[] for i in range(nodes)]
        for c, r in list(zip(col, row)):
            topo[c].append(NA3[c, r])
            if (c % 1000 == 0 & r < 10):
                print(c, r, NA3[c, r])
        for i in range(len(topo)):
            topo[i].sort(reverse=True)

        topofea = np.zeros((nodes, params['in_feature']))
        for i in range(nodes):
            for j in range(min(params['in_feature'], len(topo[i]))):
                topofea[i, j] = topo[i][j]
        np.savetxt(params["topofea_path"].format(params['in_feature']), topofea, fmt='%f')


    if os.path.exists(params['test_path']):
        test = np.loadtxt(params["test_path"], dtype = int)
        train = np.loadtxt(params["train_path"], dtype = int)
    else:
        train = np.array(range(int(nodes*args.labelrate/100)))
        test = np.array(range(int(nodes*args.labelrate/100),nodes))

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    label = torch.LongTensor(np.array(labels))
    topofea = torch.FloatTensor(topofea)
    features = torch.FloatTensor(features)
    NA = sparse_mx_to_torch_sparse_tensor(NA)

    return NA,features,topofea,label,idx_train,idx_test




















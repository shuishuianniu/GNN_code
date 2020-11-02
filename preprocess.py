import sys
sys.path.append("..")
import numpy as np

name = ["chameleon","cornell","squirrel","texas","wisconsin"]
class prepare():
    def __init__(self,name):
        with open("data/new_data/{}/out1_node_feature_label.txt".format(name)) as f:
            line = f.readlines()
        fea,lab=[],[]
        for i,l in enumerate(line):
            if i==0:
                pass
            else:
                n,f,l=l.strip().split('\t')
                fea.append(np.array(f.split(','),dtype=float))
                lab.append(int(l))
        np.savetxt("data/new_data/{}/{}.feature".format(name,name), np.array(fea),fmt='%f')
        np.savetxt("data/new_data/{}/{}.label".format(name,name),np.array(lab), fmt='%d')


        with open("data/new_data/{}/out1_graph_edges.txt".format(name)) as f:
            line = f.readlines()
            edges = []
            for i,e in enumerate(line):
                if(i!=0):
                    edges.append(np.array(e.strip().split('\t'),dtype=int))
            print(np.array(edges))
        np.savetxt("data/new_data/{}/{}.edge".format(name,name),np.array(edges),fmt="%d")


# for n in name:
# #     fea = np.genfromtxt("data/new_data/{}/{}.feature".format(n, n))
# #     label = np.genfromtxt("data/new_data/{}/{}.label".format(n, n))
# #     print(fea.shape,label)

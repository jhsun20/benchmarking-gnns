import time
import os
import pickle
import numpy as np

import dgl
import torch

import networkx as nx
from scipy import sparse as sp
import numpy as np
from data.input_features import basic_features


class load_CODataSetDGL(torch.utils.data.Dataset):
    """
    loads in a list of in the format [[networkx graph, [optimal solution1, optimal solution2, ...]], ...]

    """
    def __init__(self,
                 data_dir,
                 name,
                 split,
                 features="basic"):

        self.name = name
        self.split = split.lower()
        self.is_test = self.split == 'test'
        self.features = features
        with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        self.node_labels = []
        self.graph_lists = []
        self.graph_size = self.dataset[0][0].number_of_nodes()
        self._prepare()
        self.n_samples = len(self.graph_lists)

    def _prepare(self):

        print("preparing graphs for the %s set..." % (self.split.upper()))
        obj_values = []
        for data in self.dataset:
            nx_graph = data[0]
            labels_list = data[1]
            obj_values.append(len(labels_list[0]))

            # Create the DGL Graph
            g = dgl.from_networkx(nx_graph)

            # constant node features
            node_feat_dim = 1
            if self.features == 'basic':
                features = basic_features(nx_graph)
                node_feat_dim = 9
                g.ndata['feat'] = features
            elif self.features == "constant":
                node_feat_dim = 1
                g.ndata['feat'] = torch.zeros(g.number_of_nodes(), node_feat_dim, dtype=torch.long)

            # adding edge features for Residual Gated ConvNet
            # edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1
            g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim, dtype=torch.float)

            # change labels from list of nodes to binary
            converted_labels_list = []
            for labels in labels_list:
                temp_labels = []
                for i in range(g.number_of_nodes()):
                    if i in labels:
                        temp_labels.append(1)
                    else:
                        temp_labels.append(0)
                converted_labels_list.append(temp_labels)

            #if self.split == "test":
            if self.split == "test" or self.split == "train" or self.split == "val":
                self.graph_lists.append(g)
                while len(converted_labels_list) < 100:
                    zeros_list = [0 for _ in range(self.graph_size)]
                    converted_labels_list.append(zeros_list)
                self.node_labels.append(converted_labels_list)
            else:
                for labels in converted_labels_list:
                    self.graph_lists.append(g)
                    self.node_labels.append(labels)
                    # comment out if you want to include all solutions
                    # break

        print("Average objective value:", sum(obj_values)/len(obj_values))

        if self.split == "test":
            self.node_labels = np.array(self.node_labels)
            self.node_labels = torch.tensor(self.node_labels, dtype=torch.long)
        else:
            self.node_labels = torch.tensor(self.node_labels, dtype=torch.long)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in SBMsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return g


class CODataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, name, split, features="degree"):
        """
        split is train or test
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        self.data_dir = data_dir
        self.split = split.lower()
        self.features = features.lower()
        if self.split == "train":
            self.dataset = load_CODataSetDGL(data_dir, name, split='train', features=self.features)
        if self.split == "val":
            self.dataset = load_CODataSetDGL(data_dir, name, split='val', features=self.features)
        if self.split == "test":
            self.dataset = load_CODataSetDGL(data_dir, name, split='test', features=self.features)
        print(f"[I] Finished loading {len(self.dataset)} graphs for {self.name} dataset with {self.features} features.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        if self.split == "test":
            labels = np.array(labels)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.cat(labels).long()
        # tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        # tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        # snorm_n = torch.cat(tab_snorm_n).sqrt()
        # tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        # tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        # snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.dataset.graph_lists = [self_loop(g) for g in self.dataset.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.dataset.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.dataset.graph_lists]

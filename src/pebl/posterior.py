import math
from copy import deepcopy
from itertools import izip

import numpy as N
from pebl import network
from pebl.util import *
import pydot

class Posterior():
    def __init__(self, nodes, adjacency_matrices=[], scores=[], sorted_scored_networks=[]):
        self.nodes = nodes

        if len(adjacency_matrices) and scores:
            adjacency_matrices_and_scores = sorted(izip(adjacency_matrices, scores), cmp=lambda x,y:cmp(x[1],y[1]), reverse=True)
            adjacency_matrices, scores = unzip(adjacency_matrices_and_scores)

            self.adjacency_matrices = N.array(adjacency_matrices)
            self.scores = N.array(scores)

        elif sorted_scored_networks:
            self.adjacency_matrices = N.array([n.edgelist.adjacency_matrix for n in sorted_scored_networks])
            self.scores = N.array([n.score for n in sorted_scored_networks])

    def _consensus_matrix(self):
        norm_scores = normalize(N.exp(rescale_logvalues(self.scores)))
        return sum(n*s for n,s in zip(self.adjacency_matrices, norm_scores))

    def __iter__(self):
        for adjmat,score in zip(self.adjacency_matrices, self.scores):
            net = network.Network(self.nodes, adjmat)
            net.score = score
            yield net

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__getslice__(self, key.start, key.stop)
        
        net = network.Network(self.nodes, self.adjacency_matrices[key])
        net.score = self.scores[key]
        return net

    def __getslice__(self, i, j):
        return Posterior(self.nodes, self.adjacency_matrices[i:j], self.scores[i:j])

    def __len__(self):
        return len(self.scores)

    def consensus_network(self, threshold=.3):
        features = self._consensus_matrix()
        features[features >= threshold] = 1
        features[features < threshold] = 0
        features = features.astype(bool)
        
        return network.Network(self.nodes, features)

    @property
    def entropy(self):
        # entropy = -scores*log(scores)
        # but since scores are in log, 
        # entropy = -exp(scores)*scores
        lscores = rescale_logvalues(self.scores)
        return -N.sum(N.exp(lscores)*lscores)


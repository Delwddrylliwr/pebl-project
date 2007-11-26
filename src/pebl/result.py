import time
import socket
from bisect import insort
from copy import deepcopy

import posterior

class _ScoredNetworkList(object):
    def __init__(self, size):
        self.size = size
        self.list = []

    def push(self, scorednet):
        l = self.list
        
        if len(l) < self.size:
            if scorednet not in l:
                insort(l, deepcopy(scorednet))
        elif scorednet.score > l[0].score and scorednet not in l:
            l.remove(l[0])
            insort(l, deepcopy(scorednet))

    def __len__(self):
        return len(self.list)

    def __iter__(self):
        return self.list.__iter__()


class _ScoredNetwork(object):
    def __init__(self, edgelist, score):
        self.edgelist = edgelist
        self.score = score

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __eq__(self, other):
        return self.score == other.score and self.edgelist == other.edgelist

class LearnerResult:
    networks_to_save = 1000

    def __init__(self, learner_):
        self.data = learner_.data
        self.nodes = learner_.network.nodes
        self.networks = _ScoredNetworkList(self.networks_to_save)

    def start_run(self, learner):
        pass

    def stop_run(self):
        pass 

    def add_network(self, net, score):
        self.networks.push(_ScoredNetwork(net.edges, score))

    @property
    def posterior(self):
        return posterior.from_sorted_scored_networks(self.nodes, self.networks)


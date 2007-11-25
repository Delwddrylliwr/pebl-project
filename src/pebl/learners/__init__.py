## For our task (learning bayesian network structure) the learner is obviously the main object.
## Learners take parameters, data, priors and a network and return a (approximated) posterior 

from pebl import network
from pebl.posterior import Posterior
import  scorer
from numpy import random
import time
import socket
from pebl.util import flatten
from copy import deepcopy

class CannotAlterNetworkException(Exception):
    pass


class Learner(object):
    def __init__(self, pebldata, prior_=None, **kw):
        self.data = pebldata
        self.network = network.fromdata(pebldata)
        self.prior = prior_
        self.__dict__.update(kw)

    def _alter_network_randomly_and_score(self):
        n_nodes = self.data.variables.size

        # continue making changes and undoing them till we get an acyclic network
        for i in xrange(100):
            startnode, endnode = random.random_integers(0, n_nodes-1, 2)    
        
            if (startnode, endnode) in self.network.edges:
                # start_node -> end_node already exists, so reverse it.    
                add,remove = [(startnode, endnode)], [(endnode, startnode)]
            elif (endnode, startnode) in self.network.edges:
                # start_node <- end_node exists, so remove it
                add,remove = [], [(endnode, startnode)]
            else:
                # start_node and end_node unconnected, so connect them
                add,remove =  [(startnode, endnode)], []
            
            try:
                return self.scorer.alter_and_score_network(add=add, remove=remove)
            except scorer.CyclicNetworkError:
                continue

        # Could not find a valid network after 100 attempts  
        raise CannotAlterNetworkException() 

    def _all_changes(self):
        changes = []

        # edge removals
        changes.extend((None, edge) for edge in self.network.edges)

        # edge reversals
        reverse = lambda edge: (edge[1], edge[0])
        changes.extend((reverse(edge), edge) for edge in self.network.edges)

        # edge additions
        nz = nonzero(invert(self.network.edges.adjacency_matrix))
        changes.extend( ((src,dest), None) for src,dest in zip(*nz) )

        return changes


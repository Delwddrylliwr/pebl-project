import numpy as N

# -- Constraints
#
# The following functions return boolean functions that take an adjacency matrix as input
# and return True if the network defined by the adjacency_matrix meets the constraint.
def edge_must_exist(src, dest):
    return lambda am: am[src,dest]

def edge_mustnot_exist(src, dest):
    return lambda am: am[src,dest] == False

# -- Prior Models

class Prior(object):
    weight = 1.0

    def __init__(self, energy_matrix=None, constraints=[]):
        self.energy_matrix = energy_matrix
        self.constraints = constraints

    def log_likelihood(self, net):
        energy = N.sum(net.edges.adjacency_matrix * self.energy_matrix) 
        return -self.weight * energy

class UniformPrior(Prior):
    def __init__(self, num_nodes):
        self.energy_matrix = N.ones((num_nodes, num_nodes)) * .5
        self.constraints = []

class NullPrior(Prior):
    def __init__(self):
        pass

    def log_likelihood(self, net):
        return 0.0


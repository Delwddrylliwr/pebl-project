"""Classes for representing nodes, edges and networks.

A pebl network is a collection of nodes and directed edges between nodes.  Nodes and 
edges, while related, are unbound to each other and edges can be copied from one network 
to another with different nodes without any problems.

Edges are stored in EdgeList instances.  This module provides two implementations,
L{MatrixEdgeList} for small networks and L{SparseEdgeList} for large, sparse networks. Both
offer the same functionality with different performance characteristics.

Functions and methods accept and return nodes as numbers representing their indices and edges
as tuples of integers corresponding to (srcnode, destnode).

"""

# for randomize__OLD__()
# import as stdlib_random to avoid name clash with numpy.random
import random as stdlib_random
stdlib_random.seed()

import re
import tempfile
import tempfile
import subprocess
import os

try:
    import pydot
except:
    pass

from numpy import zeros, nonzero, all, linalg, random, identity, invert, signbit, iscomplex, any, ndarray

import data, distributions
from util import *

class EdgeList(object):    
    def __init__(self, num_nodes=0): pass

    def clear(self): 
        """Clear the list of edges."""
        pass
    
    def add(self, edge):
        """Add an edgeto the list.
        
       """
        pass
    
    def add_many(self, edges):
        for edge in edges:
            self.add(edge)

    def remove(self, edge): 
        """Remove edges from edgelist.
        
        If an edge to be removed does not exist, fail silently (no exceptions).

        """
        pass
        
    def remove_many(self, edges):
        for edge in edges:
            self.remove(edge)

    def incoming(self, node): 
        """Return list of nodes (as node indices) that have an edge to given node."""
        pass

    def outgoing(self, node): 
        """Return list of nodes (as node indices) that have an edge from given node."""
        pass

    children = outgoing
    parents = incoming
   
    def __iter__(self): 
        """Iterate through the edges in this edgelist.

        Sample usage:
        
        for edge in edgelist: print edge

        """
        pass

    @property
    def numedges(self):
        return len(list(self))

    def __contains__(self, edge):
        """Check whether an edge exists in the edgelist.


        Sample usage:
        
        if (4,5) in edgelist: print "edge exists!"

        """

        # This is the naive implementation.  
        # Subclasses should implement more efficient versions based on their
        # internal datastructures.
        for selfedge in self:
            if edge == selfedge:
                return True

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

class SparseEdgeList(EdgeList):
    """
    Maintains list of edges in two lists (for incoming and outgoing edges).

    Performance characteristics:
        - Edge insertion: O(1)
        - Edge retrieval: O(n) (*i think)
    
    If we didn't maintain two indices, retrieving an edge could take O(n^2) instead of O(n).
    
    """

    def __init__(self, num_nodes=0):
        self._outgoing = [[] for i in xrange(num_nodes)]
        self._incoming = [[] for i in xrange(num_nodes)] 

    def _resize(self, num_nodes):
        """Resize the internal indices.
        """

        self._incoming.extend([] for i in xrange(len(self._incoming), num_nodes))
        self._outgoing.extend([] for i in xrange(len(self._outgoing), num_nodes))

    def clear(self):
        self._incoming = [[] for i in xrange(len(self._incoming))]
        self._outgoing = [[] for i in xrange(len(self._outgoing))]

    def add(self, edge):
        src,dest = edge

        if dest not in self._outgoing[src]:
            self._outgoing[src].append(dest)
        if src not in self._incoming[dest]:
            self._incoming[dest].append(src)
        
    def remove(self, edge):
        src,dest = edge
        try: 
            self._incoming[dest].remove(src)
            self._outgoing[src].remove(dest)
        except ValueError: 
            pass

    def incoming(self, node):
        return self._incoming[node]

    def outgoing(self, node):
        return self._outgoing[node]

    parents = incoming
    children = outgoing

    def __iter__(self):
        for src, dests in enumerate(self._outgoing):
            for dest in dests:
                yield (src, dest)

    @property
    def adjacency_matrix(self):
        size = len(self._outgoing)
        edges = zeros((size, size), dtype=bool)
       
        selfedges = list(self)
        edges[unzip(selfedges)] = True

        return edges
    
    def __contains__(self, edge):
        src, dest = edge

        try:
            return dest in self._outgoing[src]
        except IndexError:
            return False


class MatrixEdgeList(EdgeList):
    """
    Maintains list of edges in a boolean matrix.

    Performance characteristics:
        - Edge insertion: O(1)
        - Edge retrieval: O(1)

    Memory requirements might be deemed too high for sparse networks with a large number of nodes.

    """

    def __init__(self, num_nodes=0, adjacency_matrix=None):
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        else:
            self.adjacency_matrix = zeros((num_nodes, num_nodes), dtype=bool)
        
    def _resize(self, num_nodes):
        newedges = zeros((num_nodes, num_nodes), dtype=bool)
        old_num_nodes = self.adjacency_matrix.shape[0]

        # TODO: don't do nested loops. use some numpy func to do this.
        for i in range(old_num_nodes):
            for j in range(old_num_nodes):
                newedges[i][j] = self.adjacency_matrix[i][j]

        self.adjacency_matrix = newedges

    def clear(self):
        self.adjacency_matrix = zeros(self.adjacency_matrix.shape, dtype=bool)

    def add(self, edge):
        self.adjacency_matrix[edge] = True

    def remove(self, edge):
        try: 
            self.adjacency_matrix[edge] = False
        except ValueError:
            pass # if edge does not exist, ignore error silently.

    def incoming(self, node):
        inc = nonzero(self.adjacency_matrix[:,node].flatten())
        return inc[0].tolist()

    def outgoing(self, node):
        outg = nonzero(self.adjacency_matrix[node].flatten())
        return outg[0].tolist()
    
    parents = incoming
    children = outgoing

    def __iter__(self):
        for src, dest in zip(*nonzero(self.adjacency_matrix)):
            yield (src, dest)
    
    def __contains__(self, edge): 
        try:
            return self.adjacency_matrix[edge]
        except:
            return False
    
    def __eq__(self, other):
        return self.adjacency_matrix.__hash__() == other.adjacency_matrix.__hash__()


class Network(object):
    """ A network is essentially a collection of edges between nodes.
    
    """
    
    def __init__(self, nodes):
        self.nodes = nodes
        self.edges = MatrixEdgeList(len(self.nodes))

    def is_acyclic__eigval_implementation(self):
        # first check for self-loops (1 along diagonal)
        if any(self.edges.adjacency_matrix.diagonal()):
            return False

        # network is acylcic IFF all eigenvalues of adjacency matrix are positive and real
        try:
            ev = linalg.eigvals(self.edges.adjacency_matrix)
        except:
            # the eigenvalue calculation sometimes fails to converge.. 
            return self.is_acyclic__dfs_implementation()

        if any(ev[iscomplex(ev) == True]):
            # complex eigenvalues, so not acyclic
            return False
        else:
            return not any(ev[signbit(ev) == True])

    def is_acyclic__dfs_implementation(self, startnodes=[]):
        """ 
        Checks whether the network contains any cycles/loops.
        """
        startnodes = startnodes or range(len(self.nodes))
        return self._is_acyclic(startnodes)
        
    def _is_acyclic__dfs_implementation(self, startnodes, visitednodes=[]):
        startnodes = ensure_list(startnodes)
        
        for node in startnodes:
            if node in visitednodes:
                # this node is being visited twice, therefore cycle exists
                return False
            else:
                # no cycle yet, check children
                if not self._is_acyclic__dfs_implementation(self.edges.children(node), visitednodes + [node]):
                    return False

        # got here without returning false, so no cycles below start_nodes
        return True
 
    # default implementation for is_acyclic()
    is_acyclic = is_acyclic__eigval_implementation

    def randomize__naive_implementation(self):
        n_nodes = len(self.nodes)
        density = 1.0/n_nodes

        # TODO: how big should this be?
        max_attempts = 50

        for attempt in xrange(max_attempts):
            # create an adjacency matrix with given density
            adjmat = random.rand(n_nodes, n_nodes)
            adjmat[adjmat >= (1.0-density)] = 1
            adjmat[adjmat < 1] = 0
            
            # remove self-loop edges (those along the diagonal)
            adjmat = invert(identity(n_nodes).astype(bool))*adjmat
            
            # set the adjaceny matrix and check for acyclicity
            self.edges.adjacency_matrix = adjmat.astype(bool)
            if self.is_acyclic():
                return

        # got here without finding a single acyclic network.
        # so try with a less dense network
        return self.randomize(density/2)

    # default implementation for randomize
    randomize = randomize__naive_implementation

    def layout_using_dot(self, width=400, height=400, dotpath="/Applications/Graphviz.app/Contents/MacOS/dot"): 
        tempdir = tempfile.mkdtemp(prefix="pebl")
        dot1 = os.path.join(tempdir, "1.dot")
        dot2 = os.path.join(tempdir, "2.dot")
        self.as_dotfile(dot1)

        os.system("%s -Tdot -Gratio=fill -Gdpi=60 -Gfill=10,10 %s > %s" % (dotpath, dot1, dot2))
        dotgraph = pydot.graph_from_dot_file(dot2)
      
        nodes = (n for n in dotgraph.get_node_list() if n.get_pos())
        self.node_positions = [[int(i) for i in node.get_pos().split(',')] for node in nodes] 

    layout = layout_using_dot

    def as_sparse_string(self):
        return ";".join([",".join([str(node) for node in edge]) for edge in list(self.edges)])
        
    def as_dotstring(self):
        nodename = lambda n: getattr(n, 'name', n.id)

        str = "digraph G {\n"
        for node in self.nodes:
            str += "\t\"%s\";\n" % nodename(node)

        for (src,dest) in self.edges:
            str += "\t\"%s\" -> \"%s\";\n" % (nodename(self.nodes[src]), nodename(self.nodes[dest]))
        str += "}\n"
        
        return str

    def as_dotfile(self, filename):
        f = file(filename, 'w')
        f.write(self.as_dotstring())
        f.close()

    def as_pydot(self):
        return pydot.graph_from_dot_data(self.as_dotstring())

    def as_image(self, filename, decorator=lambda x: x):
        """
        Creates an image (PNG format) for the newtork using Graphviz for layout.

        decorator is a function that accepts a pydot graph, modifies it and returns it.
        decorators can be used to set visual appearance of networks (color, style, etc).

        """

        g = self.as_pydot()
        g = decorator(g)
        g.write_png(filename, prog="dot")
        
### functions
def fromdata(data_):
    net = Network(data_.variables)
    return net

def from_nodes_and_edgelist(nodes, edgelist):
    net = Network(nodes)
    
    if isinstance(edgelist, ndarray):
        edgelist = MatrixEdgeList(adjacency_matrix=edgelist)

    net.edges = edgelist
    return net


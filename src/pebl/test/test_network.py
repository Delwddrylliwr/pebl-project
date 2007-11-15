import os
import numpy as N
from pebl import network, data
from pebl.test import datadir

class TestSparseEdgeList:
    # Which class of edgelist to use
    edgelist_class = network.SparseEdgeList

    def setUp(self):
        self.edges = self.edgelist_class(num_nodes=6)
        self.tuplelist = [(0,2), (0,5), (1,2)]
        for edge in self.tuplelist:
            self.edges.add(edge)
    
    def test_add(self):
        self.edges.add((5,1))
        assert set(self.edges) == set(self.tuplelist + [(5,1)])

    def test_add_many(self):
        self.edges.add_many([(5,1), (5,2)])
        assert set(self.edges) == set(self.tuplelist + [(5,1), (5,2)])

    def test_remove(self):
        self.edges.remove((0,2))
        assert set(self.edges) == set([(0,5), (1,2)])

    def test_remove_many(self):
        self.edges.remove_many([(0,2), (0,5)])
        assert set(self.edges) == set([(1,2)])

    def test_edgeiter(self):
        assert set(self.edges) == set(self.tuplelist), "Can use edgelist as an iterable object."
    
    def test_addedges1(self):
        self.edges.add((0, 3))
        assert (0,3) in self.edges, "Can add edges to edgelist."

    def test_incoming(self):
        assert self.edges.incoming(0) == [], "Testing edgelist.incoming"
        assert set(self.edges.incoming(2)) == set([0,1]), "Testing edgelist.incoming"
    
    def test_outgoing(self):
        assert self.edges.outgoing(2) == [], "Testing edgelist.outgoing"
        assert set(self.edges.outgoing(0)) == set([2,5]), "Testing edgelist.outgoing"
    
    def test_parents(self):
        assert self.edges.parents(0) == [], "Testing edgelist.parents"
        assert set(self.edges.parents(2)) == set([0,1]), "Testing edgelist.parents"
    
    def test_children(self):
        assert self.edges.children(2) == [], "Testing edgelist.children"
        assert set(self.edges.children(0)) == set([2,5]), "Testing edgelist.children"

    def test_contains1(self):
        assert (0,2) in self.edges, "Can check if edge in edgelist."

    def test_contains2(self):
        # Should return false without throwing exception.
        assert (99,99) not in self.edges , "Edge not in edgelist."

    def test_remove(self):
        self.edges.remove((0,2))
        assert set(self.edges) == set([(0,5), (1,2)]), "Can remove an edge."

    def test_clear(self):
        self.edges.clear()
        assert list(self.edges) == [], "Can clear edgelist."

    def test_adjacency_matrix(self):
        expected = [
            [0,0,1,0,0,1],
            [0,0,1,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
        ]
        assert (self.edges.adjacency_matrix == N.array(expected, dtype=bool)).all(), "Testing boolean matrix representation."


class TestMatrixEdgeList(TestSparseEdgeList):
    # Which class of edgelist to use
    edgelist_class = network.MatrixEdgeList

class TestNetwork:
    def setUp(self):
        self.net = network.Network([data.Variable(id=i, arity=3) for i in xrange(6)])
        for edge in [(0,1), (0,3), (1,2)]:
            self.net.edges.add(edge)

    def test_loopchecking_on_acyclic_net1(self):
        assert self.net.is_acyclic(), "Network is acyclic."

    def test_loopchecking_on_acyclic_net2(self):
        assert self.net.is_acyclic__eigval_implementation(), "Network is acyclic."

    def test_loopchecking_on_acyclic_net3(self):
        assert self.net.is_acyclic__dfs_implementation(), "Network is acyclic."

    def test_loopchecking_on_cyclic_net1(self):
        self.net.edges.add((2,0))
        assert not self.net.is_acyclic(), "Network should be cyclic."

    def test_loopchecking_on_cyclic_net2(self):
        self.net.edges.add((2,0))
        assert not self.net.is_acyclic__eigval_implementation(), "Network should be cyclic."
    
    def test_loopchecking_on_cyclic_net3(self):
        self.net.edges.add((2,0))
        assert not self.net.is_acyclic__dfs_implementation(), "Network should be cyclic."
    
    def test_randomize(self):
        self.net.randomize()
        assert self.net.is_acyclic(), "Random network should be acyclic."

    def test_randomize2(self):
        self.net.randomize__naive_implementation()
        assert self.net.is_acyclic(), "Random network should be acyclic."

    def test_as_pydot(self):
        assert len(self.net.as_pydot().edge_list) == 3, "Can convert to pydot graph instance."

    # We can only check whether the image is created or not. Cannot check if it is correct.
    def test_as_image(self):
        filename = "testnet.png"
        self.net.as_image(filename=filename)
        file_exists = filename in os.listdir(".") 
        if file_exists:
            os.remove("./" + filename)
        
        assert file_exists, "Can create image file."

class NetworkFromListOfEdges:
    def setUp(self):
        self.net = network.Netork(
            nodes = network.Network([data.Variable(id=i, arity=3) for i in xrange(6)]),
            edges = [(0,1), (4,5), (2,3)]
        )

    def test_number_of_edges(self):
        assert len(list(self.net.edges)) == 3

class NetworkFromMatrixEdgeList(NetworkFromListOfEdges):
    def setUp(self):
        edges = [(0,1), (4,5), (2,3)]
        medgelist = network.MatrixEdgeList(num_nodes=6)
        medgelist.add_many(edges)

        self.net = network.Netork(
            nodes = network.Network([data.Variable(id=i, arity=3) for i in xrange(6)]),
            edges = medgelist
        )


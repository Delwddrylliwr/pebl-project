"""Classes for learner results and statistics."""

from __future__ import with_statement

import time
import socket
from bisect import insort
from copy import deepcopy
import cPickle
import os.path

from pebl import posterior, config
from pebl.util import flatten
from pebl.network import Network

class _ScoredNetwork(Network):
    """A struct for representing scored networks.
    
    Supports comparision of networks based on score (good for sorting).
    Also supports equality checking based on first checking score equality
    (MUCH faster than checking network edges).  
    
    """

    def __init__(self, edgelist, score):
        self.edges = edgelist
        self.score = score

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __eq__(self, other):
        return self.score == other.score and self.edges == other.edges


class LearnerResult:
    """Class for storing any and all output of a learner.

    This is a mutable container for networks and scores. In the future, it will
    also be the place to collect statistics related to the learning task.

    """

    #
    # Parameters
    #
    _pformat = config.StringParameter(
        'result.format',
        'Format of the result file.',
        config.oneof('pickle'),
        default='pickle'
    )

    _pfilename = config.StringParameter(
        'result.filename',
        'The name of the result output file',
        default='result.pebl'
    )

    _psize = config.IntParameter(
        'result.numnetworks',
        """Number of top-scoring networks to save. Specify 0 to indicate that
        all scored networks should be saved.""",
        default=1000
    )

    def __init__(self, learner_=None, size=None):
        self.data = learner_.data if learner_ else None
        self.nodes = self.data.variables if self.data else None
        self.size = size or config.get('result.numnetworks')
        self.networks = []

    def start_run(self):
        """Indicates that the learner is starting a new run."""
        pass

    def stop_run(self):
        """Indicates that the learner is stopping a run."""
        pass 

    def add_network(self, net, score):
        """Add a network and score to the results."""

        nets = self.networks
        scorednet = _ScoredNetwork(net.edges, score)

        if self.size == 0 or len(nets) < self.size:
            if scorednet not in nets:
                insort(nets, deepcopy(scorednet))
        elif scorednet.score > nets[0].score and scorednet not in nets:
            nets.remove(nets[0])
            insort(nets, deepcopy(scorednet))

    def tofile(self, filename=None):
        filename = filename or config.get('result.filename')
        with open(filename, 'w') as fp:
            cPickle.dump(self, fp)

    @property
    def posterior(self):
        return posterior.from_sorted_scored_networks(
                    self.nodes, 
                    list(reversed(self.networks))
        )

def merge(*args):
    """Returns a merged result object.

    Example:
        merge(result1, result2, result3)
        results = [result1, result2, result3]
        merge(results)
        merge(*results)
    
    """
    
    results = flatten(args)
    if len(results) is 1:
        return results[0]

    # merge all networks, remove duplicates, then sort
    # TODO: can't do set(allnets) to remove dups cuz ScoredNet doesn't define a __hash__.. fix that.
    allnets = []
    for net in flatten(r.networks for r in results):
        if net not in allnets:
            allnets.append(net)
    allnets.sort()


    # create new result object
    newresults = LearnerResult()
    newresults.data = results[0].data
    newresults.nodes = results[0].nodes
    newresults.networks = allnets

    # TODO: when we add stats to LearnerResults, merge the stats here
    return newresults

def fromfile(filename):
    return cPickle.load(open(filename))

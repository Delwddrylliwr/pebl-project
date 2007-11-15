from pebl.util import *
from numpy import *
from pebl import data, distributions
import random as stdlib_random

random.seed()

# Exceptions
class CyclicNetworkError(Exception): pass

class Scorer(object):
    def __init__(self, network_, pebldata, prior_=None, subscorer=None):
        self.network = network_
        self.data = pebldata
        self.prior = prior_
        
        self.datavars = range(self.data.variables.size)
        self.score = None
        self.localscore_cache = {}
        self.subscorer = None

        if self.data.missing.any():
            self.subscorer = subscorer or MissingDataScorer

    def _globalscore(self, localscores):
        return sum(localscores)
    
    def _cpd(self, node):
        return distributions.MultinomialDistribution(
            self.data.subset(
                [node] + self.network.edges.parents(node),            # variables: node and its parents
                where(self.data.interventions[:,node] == False)[0]))  # samples: those w/o interventions on node

    def _index(self, node):
        return tuple([node] + self.network.edges.parents(node))

    def _localscore(self, node):
        index = self._index(node)
        score = self.localscore_cache.get(index, None)

        if not score:
            score = self._cpd(node).loglikelihood()
            self.localscore_cache[index] = score

        return score
    
    def _score_network_core(self):
        # use a subscorer if one exists
        if self.subscorer:
            return subscorer(self.network, self.data, self.prior).score_network()
        
        self.score = self._globalscore(self._localscore(node) for node in self.datavars)
        return self.score

    def score_network(self, net=None):
        self.network = net or self.network
        return self._score_network_core()

    def set_network(self, net):  self.network = net
    def randomize_network(self): self.network.randomize()
    def clear_network(self):     self.network.clear()


class SmartScorer(Scorer):
    def __init__(self, network_, pebldata, prior_=None, subscorer=None):
        super(SmartScorer, self).__init__(network_, pebldata, prior_, subscorer)

        self.dirtynodes = set(self.datavars)
        self.localscores = zeros((self.data.variables.size), dtype=float)
        self.last_alteration = ()
        self.saved_state = None

        # set appropriate _determine_dirtynodes() method
        if self.data.missing.any():
            self._determine_dirtynodes = self._determine_dirtynodes_with_hidden_nodes

    def _backup_state(self):
        self.saved_state = (
            self.score,                     # saved score
            self.localscores.copy()         # saved localscores
        )

    def _restore_state(self):
        if self.saved_state:
            self.score, self.localscores = self.saved_state
        
        self.saved_state = None

    def _localscore(self, node):
        localscore = super(SmartScorer, self)._localscore(node)
        self.localscores[node] = localscore
        return localscore

    def _score_network_core(self):
        # use a subscorer if one exists
        if self.subscorer:
            return subscorer(self.network, self.data, self.prior).score_network(dirtynodes=self.dirtynodes, localscores=self.localscores)

        # if no nodes are dirty, just return last score.
        if len(self.dirtynodes) == 0:
            return self.score

        # update localscore for dirtynodes, then re-calculate globalscore
        for node in self.dirtynodes:
            self.localscores[node] = self._localscore(node)
        
        self.dirtynodes = set()
        self.score = self._globalscore(self.localscores)
        return self.score
    
    def score_network(self, net=None):
        if net:
            add = [edge for edge in net.edges if edge not in self.network.edges]
            remove = [edge for edge in self.network.edges if edge not in net.edges]
        else:
            add = remove = []
        
        return self.alter_and_score_network(add, remove)
    
    def _determine_dirtynodes(self, add, remove):
        return set(unzip(add+remove, 1))

    def _determine_dirtynodes_with_hidden_nodes(self, add, remove):
        return set(self.datavars)

    def alter_and_score_network(self, add=[], remove=[]):
        """Alter the network while retaining the ability to *quickly* undo the changes."""

        self.network.edges.add_many(add)    
        self.network.edges.remove_many(remove)

        if not self.network.is_acyclic():
            self.network.edges.remove_many(add)
            self.network.edges.add_many(remove)
            raise CyclicNetworkError()
        
        # Edge src-->dest was added or removed. either way, dest's parentset changed and is thus dirty.
        self.dirtynodes = self.dirtynodes.union(self._determine_dirtynodes(add, remove))
        self.last_alteration = (add, remove)
        
        self._backup_state()
        self.score = self._score_network_core()
        
        return self.score

    def restore_network(self):
        """Undo the last alter_and_score_network()"""

        added, removed = self.last_alteration
        self._restore_state()
        
        self.dirtynodes = set()
        self.network_alterations = ()
        
        self.network.edges.add_many(removed)
        self.network.edges.remove_many(added)

    def randomize_network(self):
        self.network.randomize()
        self.dirtynodes = set(self.datavars)
    
    def set_network(self, net):
        self.network = net
        self.dirtynodes = set(self.datavars)

    def clear_network(self):
        self.network.clear()
        self.dirtynodes = set(self.datavars)


class GibbsSamplerState(object):
    """Represents the state of the Gibbs sampler.

    This state object can be used to resume the Gibbs sampler from a particaular point.
    Note that the state does not include the network or data and it's upto the caller to ensure
    that the Gibbs sampler is resumed with the same network and data.

    The following variabels are saved:
        - number of sampled scores (numscores)
        - average score (avgscore)
        - most recent value assignments for missing values (assignedvals)

    """

    def __init__(self, avgscore, numscores, assignedvals):
        autoassign(self, locals())

    @property
    def scoresum(self):
        """Log sum of scores."""
        return self.avgscore + log(self.numscores)


class MissingDataScorer(Scorer):
    """WITH MISSING DATA."""
    gibbs_burnin = 10
    
    def __init__(self, network_, pebldata, prior_=None, dirtynodes=None, localscores=None):
        super(MissingDataScorer, self).__init__(network_, pebldata, prior_)
        
        self.cpds = [None for node in self.datavars]
        self.data_dirtynodes = set(self.datavars)

        self.dirtynodes = dirtynodes or set(self.datavars)
        self.localscores = localscores or zeros((self.data.variables.size), dtype=float)

        self.subscorer = None

    def _localscore(self, node):
        self.cpds[node] = self.cpds[node] or self._cpd(node)
        return self.cpds[node].loglikelihood()

    def _score_network_core(self):
        # update localscore for data_dirtynodes, then calculate globalscore.
        for node in self.data_dirtynodes:
            self.localscores[node] = self._localscore(node)

        self.data_dirtynodes = set()
        self.score = self._globalscore(self.localscores)
        return self.score

    def _alter_data(self, row, col, value):
        oldrow = self.data.observations[row].copy()
        self.data.observations[row,col] = value

        affected_nodes = set([col] + self.network.edges.children(col))
        self.data_dirtynodes.update(affected_nodes)

        for node in affected_nodes:
            datacols = [node] + self.network.edges.parents(node)
            if not self.data.interventions[row,node]:
                self.cpds[node].replace_data(self.data.observations[row][datacols], oldrow[datacols])

    def _alter_data_and_score(self, row, col, value):
        self._alter_data(row, col, value)
        return self._score_network_core()

    def _calculate_score(self, chosenscores, gibbs_state):
        # discard the burnin period scores and average the rest
        burnin_period = self.gibbs_burnin * self.data.missing[self.data.missing==True].size

        if gibbs_state:
            # resuming from a previous gibbs run. so, no burnin required.
            scoresum = logadd(logsum(chosenscores), gibbs_state.scoresum)
            numscores = len(chosenscores) + gibbs_state.numscores
        elif len(chosenscores) > burnin_period:
            # not resuming from previous gibbs run. so remove scores from burnin period.
            nonburn_scores = chosenscores[burnin_period:]
            scoresum = logsum(nonburn_scores)
            numscores = len(nonburn_scores)
        else:
            # this occurs when gibbs iterations were less than burnin period. so use last score.
            scoresum = chosenscores[-1]
            numscores = 1
        
        score = scoresum - log(numscores)
        return score, numscores

    def score_network(self, net=None, stopping_criteria=None, gibbs_state=None, save_state=False):
        self.network = net or self.network

        # initialize cpds and data_dirtynodes
        self.cpds = [None for i in self.datavars]
        self.data_dirtynodes = set(self.datavars)
        
        # create some useful lists and local variables
        missing_indices = unzip(where(self.data.missing[:,list(self.dirtynodes)]==True))
        num_missingvals = len(missing_indices)
        arities = self.data.arities
        chosenscores = []

        # assign values to missing data 
        if gibbs_state:
            assignedvals = gibbs_state.assignedvals
        else:
            assignedvals = [stdlib_random.randint(0, arities[col]-1) for row,col in missing_indices]

        self.data.observations[unzip(missing_indices)] = assignedvals

        # score once to set cpds and localscores
        self._score_network_core()

        # default stopping criteria is to sample for N^2 iterations (N == number of missing vals)
        stopping_criteria = stopping_criteria or (lambda scores,iterations,N: iterations >= num_missingvals**2)

        # Gibbs Sampling: 
        # For each missing value:
        #    1) score net with each possible value (based on node's arity)
        #    2) using a probability wheel, sample a value from the possible values
        iterations = 0
        while not stopping_criteria(chosenscores, iterations, num_missingvals):
            for row,col in missing_indices:
                scores = [self._alter_data_and_score(row, col, val) for val in xrange(arities[col])]
                chosenval = logscale_probwheel(scores)
                self._alter_data(row, col, chosenval)
                chosenscores.append(scores[chosenval])
            
            iterations += num_missingvals

        chosenscores = array(chosenscores)
        self.score, numscores = self._calculate_score(chosenscores, gibbs_state)

        self.chosenscores = chosenscores

        # save state of gibbs sampler?
        if save_state:
            self.gibbs_state = GibbsSamplerState(
                    avgscore=self.score, 
                    numscores=numscores, 
                    assignedvals=self.data.observations[unzip(missing_indices)].tolist())

        return self.score


class MissingDataExactScorer(MissingDataScorer):
    def score_network(self, net=None):
        self.network = net or self.network
        
        # initialize cpds and data_dirtynodes
        self.cpds = [None for i in self.datavars]
        self.data_dirtynodes = set(self.datavars)

        # create some useful lists and local variables
        missing_indices = unzip(where(self.data.missing[:,self.dirtynodes]==True))
        num_missingvals = len(missing_indices)
        possiblevals = [range(self.data.variables[col].arity) for row,col in missing_indices]

        # score once to set cpds and localscores
        self._score_network_core()
        
        scores = []
        info = []
        for assignedvals in cartesian_product(possiblevals):
            # set missingvals to assignedvals
            for (row,col),val in zip(missingvals, assignedvals):
                self._alter_data(row, col, val)

            score = self._score_network_core()
            scores.append(score)
            info.append((assignedvals, entropy_of_list(assignedvals), self.localscores.tolist(), score))

        self.score = logsum(scores) - log(len(scores))
        self.scores = scores
        self.info = info

        return self.score

class MaximumEntropyMissingDataScorer(MissingDataScorer):
    def _do_maximum_entropy_assignment(self, var):
        """Assign values to the missing values for this variable such that
        it has a maximum entropy discretization.
        """

        arity = self.data.variables[var].arity
        numsamples = self.data.samples.size

        missingvals = self.data.missing[:,var]
        missingsamples = where(missingvals == True)[0]
        observedsamples = where(missingvals == False)[0]
        
        # maximum entropy discretization for *all* samples for this variable
        numeach = numsamples/arity
        assignments = flatten([val]*numeach for val in xrange(arity))
        for i in xrange(numsamples - len(assignments)):  
            assignments.append(i)

        # remove the values of the observed samples
        for val in self.data.observations[observedsamples, var]:
            assignments.remove(val)

        random.shuffle(assignments)
        self.data.observations[missingsamples,var] = assignments

    def _swap_data(self, var, sample1, choices_for_sample2):
        val1 = self.data.observations[sample1, var]
        
        # try swapping till we get a different value (but don't keep trying forever)
        for i in xrange(len(choices_for_sample2)/2):
            sample2 = stdlib_random.choice(choices_for_sample2)
            val2 = self.data.observations[sample2, var]
            if val1 != val2:
                break

        self._alter_data(sample1, var, val2)
        self._alter_data(sample2, var, val1)
        
        return (sample1, var, val1, sample2, var, val2)
    
    def _undo_swap(self, row1, col1, val1, row2, col2, val2):
        self._alter_data(row1, col1, val1)
        self._alter_data(row2, col2, val2) 

    def score_network(self, stopping_criteria=None, gibbs_state=None, save_state=False):
        # network was altered.. so reset cpds and data_dirtynodes
        self.cpds = [None for i in self.datavars]
        self.data_dirtynodes = set(self.datavars)

        # create some useful lists and vars
        chosenscores = []
        numsamples = self.data.samples.size
        num_missingvals = self.data.missing[self.data.missing == True].shape[0]
        
        # any var not fully observed (with at least one missing value)
        missingvars = [v for v in self.datavars if
                                v in self.dirtynodes and 
                                self.data.missing[:,v].any()]

        missingsamples = [where(self.data.missing[:,var] == True)[0] for var in self.datavars]

        # set missing values using last assigned values from previous gibbs run or random values based on node arity
        if gibbs_state:
            assignedvals = gibbs_state.assignedvals
            self.data.observations[where(self.data.missing==True)] = assignedvals
        else:
            for var in missingvars:
                self._do_maximum_entropy_assignment(var)
        
        # score to initialize cpds, etc.
        self._score_network_core()

        # default stopping criteria is to sample for N^2 iterations (N == number of missing vals)
        stopping_criteria = stopping_criteria or (lambda scores,iterations,N: iterations >= num_missingvals**2)

        # === Gibbs Sampling === 
        # For each missing value:
        #    1) score net with each possible value (based on node's arity)
        #    2) using a probability wheel, sample a value from the possible values (and set it in the dataset)
        iterations = 0
        while not stopping_criteria(chosenscores, iterations, num_missingvals):
            for var in missingvars:  
                for sample in missingsamples[var]:
                    score0 = self._score_network_core()
                    swap = self._swap_data(var, sample, missingsamples[var]) 
                    score1 = self._score_network_core() 
                    chosenval = logscale_probwheel([score0, score1])
                    
                    if chosenval == 0:
                        # undo swap and select old_score
                        self._undo_swap(*swap)
                        chosenscores.append(score0)
                    else:
                        chosenscores.append(score1)

                iterations += numsamples

        chosenscores = array(chosenscores)
        self.score, numscores = self._calculate_score(chosenscores, gibbs_state)

        self.chosenscores = chosenscores

        # save state of gibbs sampler?
        if save_state:
            self.gibbs_state = GibbsSamplerState(
                    avgscore=self.score, 
                    numscores=numscores, 
                    assignedvals=self.data.observations[where(self.data.missing==True)].tolist())
        
        return self.score




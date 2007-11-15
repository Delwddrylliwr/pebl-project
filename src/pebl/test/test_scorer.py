# set numpy.test to None so we don't run numpy's tests.
from numpy import *
test = None

from pebl.learners import scorer
from pebl import data, network
from pebl.test import datadir
import os
import copy

def _create_data_and_net():
    # this data and net from test_stochfunc.TestMultinomialStochasticFunction.test_loglikelihood
    a = array([[0, 1, 1, 0],
               [1, 0, 0, 1],
               [1, 1, 1, 0],
               [1, 1, 1, 0],
               [0, 0, 1, 1]
        ])
    dat = data.PeblData(a)
    net = network.fromdata(dat)
    net.edges.add_many([(1,0), (2,0), (3,0)])
    
    return dat, net


class TestBasicScorer:
    def setUp(self):
        self.data, self.net = _create_data_and_net()
        self.scorer = scorer.Scorer(self.net, self.data)

    def test_create_distribution(self):
        try:
            self.scorer._cpd(0)
            self.scorer._cpd(1)
        except:
            assert False, "Trying to create distributions."

    def test_score_node(self):
        score = self.scorer._localscore(0) 
        assert score - -3.87120101107 <= .0001, "Checking for correct score." 
        
    def test_caching(self):
        # score node, then check if it's in cache.
        self.scorer._localscore(0)
        assert self.scorer._index(0) in self.scorer.localscore_cache.keys(), "Checking cache."

class TestSmartScorer:
    def setUp(self):
        self.data, self.net = _create_data_and_net()
        self.scorer = scorer.SmartScorer(self.net, self.data)

        self.net2 = network.fromdata(self.data)
        self.net2.edges.add_many(list(self.net.edges))

    def test_alteration_cyclic(self):
        try:
            self.scorer.alter_and_score_network(add=[(0,1)]), "Creating cyclic network."
            assert False, "Should've raised exception on cyclic network"
        except scorer.CyclicNetworkError:
            pass

    def test_alteration_cyclic2(self):
        try:
            self.scorer.alter_and_score_network(add=[(0,1)])
        except scorer.CyclicNetworkError:
            pass

        assert (0,1) not in self.net.edges, "Network shouldn't be altered to be in cyclic state."

    def test_alteration_scoring(self):
        # scorer(net2).score_network()
        #  VS
        # scorer(net1).alter_network() then, scorer.score_network()
        # scores should be the same.
        score1 = self.scorer.alter_and_score_network(add=[(2,3)])

        self.net2.edges.add((2,3))
        scorer2 = scorer.Scorer(self.net2, self.data)
        score2 = scorer2.score_network()

        assert score1 == score2, "Scores should be same regardless of how we arrive at network."

    def test_restoring1(self):
        # score net, make changes, undo changes, scores should be same
        score1 = self.scorer.score_network()
        saved_localscores = copy.deepcopy(self.scorer.localscores)

        score2 = self.scorer.alter_and_score_network(add=[(2,3), (1,2)], remove=[(1,0)])
        self.scorer.restore_network()
        score3 = self.scorer.score_network()

        assert score1 == score3, "Alter followed by restore should leave score unchanged."
        assert all(self.scorer.localscores == saved_localscores)

    def test_scoring(self):
        # ensure that score is correct after a series of alters and restores.
        score1 = self.scorer.score_network()
        self.scorer.alter_and_score_network(add=[(2,3)])
        self.scorer.alter_and_score_network(remove=[(1,0)])
        self.scorer.alter_and_score_network(add=[(1,2)])
        self.scorer.restore_network()
        self.scorer.alter_and_score_network(add=[(1,0)])
        self.scorer.alter_and_score_network(remove=[(2,3)])
        score2 = self.scorer.score_network()
        
        assert score1 == score2, "Long series of alters and restores."


class TestMissingDataScorer:
    scorer_type = scorer.MissingDataScorer

    def setUp(self):
        # write out test data (don't want to depend on external data files)
        a,b,c,d,e = 0,1,2,3,4
        
        self.data = data.fromfile(os.path.join(datadir(), 'testdata5.txt'))
        self.net = network.fromdata(self.data)
        self.net.edges.add_many([(a,c), (b,c), (c,d), (c,e)])
        self.scorer = self.scorer_type(self.net, self.data)

    def test_gibbs_scoring(self):
        # score two nets (correct one and bad one) with missing values.
        # ensure that correct one scores better. (can't check for exact score)
        stopping_criteria = lambda scores,iterations,N: iterations >= 10*N**2


        # score network: {a,b}->c->{d,e}
        score1 = self.scorer.score_network(stopping_criteria=stopping_criteria)

        # score network: {a,b}->{d,e} c
        a,b,c,d,e = 0,1,2,3,4

        data2 = data.fromfile(os.path.join(datadir(), 'testdata5.txt'))
        net2 = network.fromdata(self.data)
        net2.edges.add_many([(a,d), (a,e), (b,d), (b,e)])
        scorer2 = self.scorer_type(net2, data2)
        score2 = scorer2.score_network(stopping_criteria=stopping_criteria)

        # score1 should be better than score2
        print score1, score2
        assert score1 > score2, "Gibbs sampling can find goodhidden node."

    def test_gibbs_saving_state(self):
        score1 = self.scorer.score_network(save_state=True)
        gibbs_state = self.scorer.gibbs_state
        
        assert len(gibbs_state.assignedvals) == len(self.data.missingvals), "Can save state of Gibbs sampler."

    def test_gibbs_restoring_state(self):
        score1 = self.scorer.score_network(save_state=True)
        gibbs_state = self.scorer.gibbs_state
        score2 = self.scorer.score_network(gibbs_state=gibbs_state, stopping_criteria=lambda scores,numscores,n: numscores>0)
        
        # no way to check if score is correct but we can check whether everything proceeds without any errors.
        assert True

    def test_alterdata_dirtynodes(self):
        # alter data. check dirtynodes.
        self.scorer.score_network()   # to initialize datastructures
        self.scorer._alter_data(0, 2, 1, self.data.interventions)
        assert set(self.scorer.data_dirtynodes) == set([0,1,2,3,4]), "Altering data dirties affected nodes."

    def test_alterdata_scoring(self):
        # score. alter data. score. alter data (back to original). score. 
        # 1st and last scores should be same.
        self.scorer.score_network()
        score1 = self.scorer._score_network()
        oldval = self.scorer.data[0][2]
        self.scorer._alter_data(0, 2, 1, self.data.interventions)
        self.scorer._score_network()
        self.scorer._alter_data(0, 2, oldval, self.data.interventions)
        score2 = self.scorer._score_network()

        assert score1 == score2, "Altering and unaltering data leaves score unchanged."

    def test_restore_network(self):
        # score net. alter net. restore net. score. 
        # scores should be the *exact* same.
        a,b,c,d,e = 0,1,2,3,4
        score1 = self.scorer.score_network()
        self.scorer.alter_network(add=[(b,e)])
        self.scorer.restore_network()
        score2 = self.scorer.score_network()

        assert score1 == score2, "Can restore network alterations."

    def test_restore_network2(self):
        # score net. alter net. *score*. restore net. score. 
        # scores should be the *exact* same.
        a,b,c,d,e = 0,1,2,3,4
        score1 = self.scorer.score_network()
        self.scorer.alter_network(add=[(b,e)])
        self.scorer.score_network()
        self.scorer.restore_network()
        score2 = self.scorer.score_network()

        assert score1 == score2, "Can restore network alterations (even after it's been scored)."

class TestMaximumEntropyMissingDataScorer(TestMissingDataScorer):
    scorer_type = scorer.MaximumEntropyMissingDataScorer

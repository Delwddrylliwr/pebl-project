"""Classes and functions for Simulated Annealing learner"""

from math import exp
from pebl import network, result, evaluator, config
from pebl.learner import Learner


class SALearnerStatistics:
    def __init__(self, starting_temp, delta_temp, max_iterations_at_temp):
        self.temp = starting_temp
        self.iterations_at_temp = 0
        self.max_iterations_at_temp = max_iterations_at_temp
        self.delta_temp = delta_temp

        self.iterations = 0
        self.best_score = 0
        self.current_score = 0

    def update(self):
        self.iterations += 1
        self.iterations_at_temp += 1

        if self.iterations_at_temp >= self.max_iterations_at_temp:
            self.temp *= self.delta_temp
            self.iterations_at_temp = 0


class SimulatedAnnealingLearner(Learner):
    #
    # Parameters
    #
    _pstart_temp = config.FloatParameter(
        'simanneal.start_temp',
        "Starting temperature for a run.",
        config.atleast(0.0),
        default=100.0
    )

    _pdelta_temp = config.FloatParameter(
        'simanneal.delta_temp',
        'Change in temp between steps.',
        config.atleast(0.0),
        default=0.5
    )

    _pmax_iters = config.IntParameter(
        'simanneal.max_iters_at_temp',
        'Max iterations at any temperature.',
        config.atleast(0),
        default=100
    )

    _pseed = config.StringParameter(
        'simanneal.seed',
        'Starting network for a greedy search.',
        default=''
    )

    def __init__(self, data_=None, prior_=None, start_temp=None,
                 delta_temp=None, max_iters_at_temp=None, seed=None):
        """Create a Simulated Aneaaling learner.

        start_temp, delta_temp, max_iters_at_temp and seed can be specified via
        the constructor or configuration parameters.
       
        """

        super(SimulatedAnnealingLearner,self).__init__(data_, prior_)
        self.start_temp = start_temp or config.get('simanneal.start_temp')
        self.delta_temp = delta_temp or config.get('simanneal.delta_temp')
        self.max_iters_at_temp = max_iters_at_temp or \
                                 config.get('simanneal.max_iters_at_temp')

        self.seed = seed or \
                    network.Network(self.data.variables,
                                    config.get('simanneal.seed'))


    def run(self):
        """Run the learner."""

        self.stats = SALearnerStatistics(self.start_temp, self.delta_temp, 
                                         self.max_iters_at_temp)
        self.result =  result.LearnerResult(self)
        self.evaluator = evaluator.fromconfig(self.data, self.seed, self.prior)
        self.evaluator.set_network(self.seed.copy())

        self.result.start_run()
        curscore = self.evaluator.score_network()
        
        # temperature decays exponentially, so we'll never get to 0. So, we continue until temp < 1
        while self.stats.temp >= 1:
            try:
                self._alter_network_randomly()
            except CannotAlterNetworkException:
                return

            newscore = self.evaluator.score_network()
            self.result.add_network(self.evaluator.network, newscore)

            if self._accept(newscore):
                # set current score
                self.stats.current_score = newscore
                if self.stats.current_score > self.stats.best_score:
                    self.stats.best_score = self.stats.current_score
            else:
                # undo network alteration
                self.evaluator.restore_network()

            # temp not updated EVERY iteration. just whenever criteria met.
            self.stats.update() 

        self.result.stop_run()
        return self.result

    def _accept(self, newscore):
        oldscore = self.stats.current_score

        if newscore >= oldscore:
            return True
        elif random.random() < exp((newscore - oldscore)/self.stats.temp):
            return True
        else:
            return False


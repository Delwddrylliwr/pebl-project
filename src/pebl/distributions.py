import math
from util import ensure_list
from numpy import *
from itertools import izip

class Distribution(object):
    def __init__(self, pebldata): pass 
    def loglikelihood(self): pass

#############################################################################################################
class MultinomialDistribution(Distribution):
    __title__ = "Multinomial Distribution"
    lnfactorial_cache = array([])

    def __init__(self, pebldata):
        self.data = pebldata
        arities = pebldata.arities

        max_count = pebldata.samples.size + max(arities)
        if len(self.lnfactorial_cache) < max_count:
            self.prefill_lnfactorial_cache(max_count)
        
        # create a Conditional Probability Table
        qi = int(product(arities[1:]))
        self.counts = zeros((qi, arities[0] + 1), dtype=int)
        
        if pebldata.variables.size == 1:
            self.offsets = [0]
        else:
            multipliers = concatenate(([1], pebldata.arities[1:-1]))
            offsets = multiply.accumulate(multipliers)
            self.offsets = concatenate(([0], offsets))

        # add data to cpt
        self._change_counts(pebldata.observations, 1)

    def _change_counts(self, observations, change=1):
        indices = dot(observations, self.offsets)
        child_values = observations[:,0]

        for j,k in izip(indices, child_values):
            self.counts[j][k] += change
            self.counts[j][-1] += change

    def replace_data(self, add, remove):
        add_index = sum(i*o for i,o in izip(add, self.offsets))
        remove_index = sum(i*o for i,o in izip(remove, self.offsets))

        self.counts[add_index][add[0]] += 1
        self.counts[add_index][-1] += 1

        self.counts[remove_index][remove[0]] -= 1
        self.counts[remove_index][-1] -= 1

    def loglikelihood(self):
        lnfac = self.lnfactorial_cache
        counts = self.counts

        ri = self.data.arities[0]
        
        result = sum( 
              lnfac[ri-1]                           # log((ri-1)!) 
            - lnfac[counts[:,-1] + ri -1]           # log((Nij + ri -1)!)
            + sum(lnfac[counts[:,:-1]], axis=1)     # log(Product(Nijk!)) == Sum(log(Nijk!))
        )

        return result


    def prefill_lnfactorial_cache(self, size):
        # logs = log(x) for x in [0, 1, 2, ..., size+10]
        #    * EXCEPT, log(0) = 0 instead of -inf.
        logs = concatenate(([0.0], log(arange(1, size+10, dtype=float))))

        # add.accumulate does running sums..
        MultinomialDistribution.lnfactorial_cache = add.accumulate(logs)


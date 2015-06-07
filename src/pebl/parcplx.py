"""Classes for fNML local complexity penalty"""

import math
import itertools
import numpy as N

try:
    from pebl import _parcplx
except:
    _parcplx = None

#
# Complexity Penalty classes
#
class ParCplx(object):
    """Parametric complexity.
    
    @TODO both a pure-python and a fast C implementation. The C implementation will be
    used if available.
    
    """
    def __init__(self, data_):
        """Create a CPD.
        data_ should only contain data for the nodes involved in this CPD. The
        first column should be for the child node and the rest for its parents.
        
        The Dataset.subset method can be used to create the required dataset::
            d = data.fromfile("somedata.txt")
            n = network.random_network(d.variables)
            d.subset([child] + n.edges.parents(child))
        """

    def complexity(self):
        """Calculates the parametric complexity for the bayesian network, 
        in line with factorised normalised maximum likelihood.
        
        """ 
        pass


class Cplx_Py(ParCplx):
    """Pure python implementation of Parametric Complexity.
    
       Because the multinomial parametric complexity is calculated iteratively, 
       we need to run this once on initialisation, to build a table of multinomial
       complexities, and then return the suitable one
    """

    # caches shared by all instances
    factorial_cache = N.array([[]])
    multinomial_cache = N.array([[]])

    def __init__(self, data_):
        self.data = data_
        arities = [v.arity for v in data_.variables]

        # ensure that there won't be a cache miss when using the global table
        #of multinomial complexities, by finding the biggest sample size needed
        #...construct an ordering of the possible combinations of values
        #for the parents...
        #first, create a list of the unique values for each variable
        def uniquify_sort(seq):
            set = {}
            map(set.__setitem, seq, [])
            return set.keys().sort
        possible_values = map(uniquify, data.observations)
        #now construct a list of all combinations of possible values for 
        #each variable
        list_product = list(itertools.product(possible_values))
        
        #we now need the frequency with which each of these combinations appears
        #in the data
#        def combination_freq(seq):
#            return data.observations.count(seq)
#        sample_sizes = map(combination_freq, list_product)
        sample_sizes = map(data.observations.count, list_product)
        
        #finally we'll check if we already have enough complexities cached
        if ( len(self.__class__.multinomial_cache) < arities[0] 
        or len(self.__class__.multinomial_cache[0]) < max(sample_sizes) ):
            self._prefill_factorial_cache(maxcount)


    #
    # Public methods
    #
    def complexity(self):
        mn_cplx = self.multinomial_cache
        
        num_parents = length(self.data.variables)
        
        #now we want the log-multinomial complexity for each of these observed 
        #combinations
        def observed_complexity(freq):
            return mn_cplx[arity-1][freq]
        result = N.sum(N.log(map(observed_complexity, sample_sizes)))

        return result

    #
    # Private methods
    #
    def _change_counts(self, observations, change=1):
        indices = N.dot(observations, self.offsets)
        child_values = observations[:,0]

        for j,k in izip(indices, child_values):
            self.counts[j,k] += change
            self.counts[j,-1] += change

    def _prefill_multinomial_cache(self, sample_size, arity):
        #we want to preserve the existing complexities, and only calculate new
        mn_cache = self.__class__.multinomial_cache
        
        #we'll need factorials up to the sample size
        # ensure that there won't be a cache miss
        if len(self.__class__.factorial_cache) < sample_size:
            self._prefill_factorial_cache(sample_size)
    
        #c^1_n is 1 for all n
        for n in range(len(mn_cache[1]), sample_size):
            mn_cache[0][n] = 1
        
        #c^2_n is more complex
        for n in range(len(mn_cache[1]), sample_size):
            def complexity_term(h):
                fctl = __class__.factorial_cache
                return (fctl[n] / (fctl[h] * fctl[n-h])) * pow(h/n, h) * pow((n-h)/n, n-h)
            mn_cache[1][n] = N.add.accumulate(map(complexity_term, range(0, n)))
        
        #each existing arity needs bringing up to sample size
        for r in range(2, len(mn_cache)):
            for n in range(len(mn_cache[r]), sample_size):
                #because r = arity-1, arity - 2 = r-1...
                mn_cache[r][n] = mn_cache[r-1][n] + ( n / (r-1)) * mn_cache[r-2][n]
        
        #finally, each new arity needs constructing from scratch
        for r in range(len(mn_cache), arity + 1):
            mn_cache[r][0] = 1
            
            for n in range(1, sample_size):
                #because r = arity-1, arity - 2 = r-1...
                mn_cache[r][n] = mn_cache[r-1][n] + ( n / (r-1)) * mn_cache[r-2][n]

        
    def _prefill_factorial_cache(self, size):
        # logs = log(x) for x in [0, 1, 2, ..., size+10]
        #    * EXCEPT, log(0) = 0 instead of -inf.
        numbers = N.concatenate(([0.0], N.arange(1, size+10, dtype=float)))

        # add.accumulate does running sums..
        self.__class__.factorial_cache = N.multiply.accumulate(numbers)


#class Cplx_C(MultinomialCPD_Py):
#    """C implementation of parametric complexity."""
#
#    def __init__(self, data_):
#        if not _cpd:
#            raise Exception("_cpd C extension module not loaded.")
#
#        self.data = data_
#        arities = [v.arity for v in data_.variables]
#        num_parents = len(arities)-1
#
#        # ensure that there won't be a cache miss
#        maxcount = data_.samples.size + max(arities)
#        if len(self.__class__.lnfactorial_cache) < maxcount:
#            self._prefill_lnfactorial_cache(maxcount)
#        
#        self.__cpt = _cpd.buildcpt(data_.observations, arities, num_parents)
#
#    def loglikelihood(self):
#        return _cpd.loglikelihood(self.__cpt, self.lnfactorial_cache)
#
#    def replace_data(self, oldrow, newrow):
#        _cpd.replace_data(self.__cpt, oldrow, newrow)
#
#    def __del__(self):
#        _cpd.dealloc_cpt(self.__cpt)
#
#
## use the C implementation if possible, else the python one
#MultinomialCPD = MultinomialCPD_C if _cpd else MultinomialCPD_Py


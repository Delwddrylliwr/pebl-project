import os
from operator import attrgetter
import numpy as N
from pebl import data
from pebl.test import datadir

TESTDATA1 = """samples	var 1	var 2	var 3
sample 1	2.5!	!X	1.7
sample 2	1.1	!1.7	2.3
sample 3	4.2	999.3	12
"""

class TestBasicFileParsing:
    def setUp(self):
        self.data = data.fromfile(os.path.join(datadir(), "testdata1.txt"))        

    def test_value_parsing(self):
        assert self.data.observations[0,0] == 2.5, "Parsing values."
        assert self.data.observations[0][2] == 1.7, "Parsing values."
        assert self.data.observations[2,1] == 999.3, "Parsing values."
        assert self.data.observations[2][2] == 12.0, "Parsing values: convert to float."

    def test_interventions(self):
        assert self.data.interventions[0,0] == True, "Parsing interventions (! after value.)"
        assert self.data.interventions[1,1] == True, "Parsing interventions (! before value.)"
        assert self.data.interventions[0].tolist() == [True, True, False]
        assert N.where(self.data.interventions[0] == True)[0].tolist() == [0,1]

    def test_missingvals(self):
        assert self.data.missing[0,0] == False, "Parsing missingvals."
        assert self.data.missing[0,1] == True, "Parsing missingvals."

    def test_interventions_and_missingvals(self):
        result1 = self.data.missing[0,1] 
        result2 = N.where(self.data.interventions[0] == True)[0].tolist() 
        assert (result1 == True and result2 == [0, 1]), "Parsing both interventions and missingvals (! and X) for the same value."


class TestAdvancedFileParsing1:
    def setUp(self):
        self.data = data.fromfile(os.path.join(datadir(), "testdata2.txt"))        

    def test_value_parsing(self):
        o = self.data.observations

        assert o[0,0] == 0
        assert o[1,1] == 1
        assert o[2,1] == 2

    def test_resulting_dtype(self):
        assert self.data.observations.dtype == N.dtype('byte')

    def test_variable_names(self):
        assert self.data.variables[0].name == 'shh'
        assert [v.name for v in self.data.variables] == ['shh', 'ptchp', 'smo']

    def test_sample_names(self):
        assert self.data.samples[0].name == 'mouse1'
        assert [s.name for s in self.data.samples] == ['mouse1', 'mouse2', 'mouse3']

    def test_variable_int_attr(self):
        assert [v.arity for v in self.data.variables] == [2,3,2]

    def test_variable_bool_attr(self):
        assert [v.protein for v in self.data.variables] == [False, True, False]

    def test_variable_filtering(self):
        assert map(lambda v: v.protein, self.data.variables) == [False, True, False] 
        assert filter(None, map(lambda v: v.name if v.protein else None, self.data.variables)) == ['ptchp']

    def test_variables_where(self):
        assert self.data.variables_where(lambda v: v.protein) == [1]
        assert self.data.observations[:,self.data.variables_where(lambda v: not v.protein)].tolist() == [[0,1], [1,1], [1,0]]

    def test_samples_where(self):
        assert self.data.samples_where(lambda s: s.name.endswith('1')) == [0]

    def test_auto_variable_id(self):
        assert self.data.variables[2].id == '2'


class TestAdvancedFileParsing2:
    def setUp(self):
        # the file has extra whitespaces (which shouldn't break the parser)
        self.data = data.fromfile(os.path.join(datadir(), "testdata3.txt"))        

    def test_auto_arity(self):
        assert [v.arity for v in self.data.variables] == [2,3,2]
        assert self.data.arities == [2,3,2]

    def test_variable_names(self):
        assert self.data.variables[0].name == 'shh'
        assert [v.name for v in self.data.variables] == ['shh', 'ptchp', 'smo']


def _create_data_mockup(full=True):
    a = N.array([[1.2, 1.4, 2.1, 2.2, 1.1],
                 [2.3, 1.1, 2.1, 3.2, 1.3],
                 [3.2, 0.0, 2.2, 2.5, 1.6],
                 [4.2, 2.4, 3.2, 2.1, 2.8],
                 [2.7, 1.5, 0.0, 1.5, 1.1],
                 [1.1, 2.3, 2.1, 1.7, 3.2] ])

    interventions = N.array([[0,0,0,0,0],
                             [0,1,0,0,0],
                             [0,0,1,1,0],
                             [0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,0,0,1,0] ])

    missing = N.array([[0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,1,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,0,0] ])
    
    variablenames = ["gene A", "gene B", "receptor protein C", " receptor D", "E kinase protein"]
    samplenames = ["head.wt", "limb.wt", "head.shh_knockout", "head.gli_knockout", 
                   "limb.shh_knockout", "limb.gli_knockout"]
    
    if full:
        return data.PeblData(
            observations = a,
            missing = missing,
            interventions = interventions,
            variables = N.array([data.Variable(id=str(i), name=n) for i,n in enumerate(variablenames)]),
            samples = N.array([data.Sample(id=str(i), name=n) for i,n in enumerate(samplenames)])
        )
    else:
        return data.PeblData(observations=a)


class TestDataObject:
    """ Test PeblData's methods and properties.
        
        We're not testing parsing here -- so we build the object manually.
    
    """
    
    def setUp(self):
        self.data = _create_data_mockup()

    def test_numvariables(self):
        assert self.data.variables.size == 5, "Data has 5 variables."

    def test_numsamples(self):
        assert self.data.samples.size == 6, "Data has 6 samples."

    def test_subsetting_byvar(self):
        assert (self.data.subset(variables=[0,2,4]).observations == self.data.observations.take([0,2,4], axis=1)).all(), "Subsetting data by variables."
        
    def test_subsetting_bysample(self):
        assert (self.data.subset(samples=[0,2]).observations == self.data.observations.take([0,2], axis=0)).all(), "Subsetting data by samples."
    
    def test_subsetting_byboth(self):
        assert (self.data.subset(variables=[0,2], samples=[1,2]).observations == self.data.observations[N.ix_([1,2],[0,2])]).all(), "Subsetting data by variable and sample."

    def test_subsetting_missingvals(self):
        subset = self.data.subset(variables=[1,2], samples=[2,3,4])
        assert (subset.missing == self.data.missing[[2,3,4]][:,[1,2]]).all(), "Missingvals in data subset."

    def test_subsetting_interventions(self):
        subset = self.data.subset(variables=[1,2], samples=[2,3,4])
        assert (subset.interventions == self.data.interventions[[2,3,4]][:,[1,2]]).all(), "Interventions in data subset."

    def test_subsetting_variableinfo(self):
        subset = self.data.subset(variables=[1,2], samples=[2,3,4])
        assert [v.name for v in subset.variables] == ["gene B", "receptor protein C"], "Variable names in data subset."

    def test_subsetting_samplenames(self):
        subset = self.data.subset(variables=[1,2], samples=[2,3,4])
        assert [s.name for s in subset.samples] == ["head.shh_knockout", "head.gli_knockout", "limb.shh_knockout"], "Sample names in data subset."

    def test_has_missing_values(self):
        assert self.data.missing.any(), "Check for missing values."

    def test_missingvals_indices(self):
        missing = N.transpose(N.where(self.data.missing)).tolist()
        assert missing == [[2,1], [3,1], [4,2]], "Missing mask as (row,col) indices."

    def test_variables_where(self):
        assert self.data.variables_where(lambda v: v.name.find("protein") != -1) == [2,4]

    def test_samples_where(self):
        assert self.data.samples_where(lambda s: s.name.startswith('head')) == [0,2,3]

class TestDataCreation:
    def setUp(self):
        self.data = _create_data_mockup(full=False)

    def test_interventions(self):
        assert self.data.interventions.shape == self.data.observations.shape
        assert self.data.interventions.any() == False

    def test_missing(self):
        assert self.data.missing.shape == self.data.observations.shape
        assert self.data.missing.any() == False

    def test_variables(self):
        assert self.data.variables.size == self.data.observations.shape[1]
        assert self.data.variables[0].id == '0'

    def test_samples(self):
        assert self.data.samples.size == self.data.observations.shape[0]
        assert self.data.samples[0].id == '0'

class TestDiscretizing:
    """Test data discretization."""

    def setUp(self):
        self.data = data.fromfile(os.path.join(datadir(), 'testdata4.txt'))

    def test_basic_discretizing(self):
        newdata = data.discretize_variables(self.data)
        assert newdata.observations[:,1].tolist() == [1, 0, 0, 2, 1, 2, 0, 2, 1], "Discretizing without any parameters."

    def test_resulting_arities(self):
        newdata = data.discretize_variables(self.data, numbins=3)
        assert newdata.arities[2] == 3, "Arities of discretized data."

    def test_discretizing_with_many_equal_values(self):
        newdata = data.discretize_variables(self.data, numbins=3)
        assert newdata.observations[:,4].tolist() == [0, 1, 2, 2, 0, 2, 0, 0, 0], "Discretizing with many equal values."

    def test_includevars(self):
        newdata = data.discretize_variables(self.data, numbins=3, includevars=[0,2])
        assert newdata.observations[:,1].tolist() == self.data.observations[:,1].tolist(), "Don't discretize variable if not in includevars."
        assert newdata.observations[:,2].tolist() == [1, 1, 0, 2, 0, 1, 2, 0, 2], "Discretize variable if in includevars."

    def test_excludevars(self):
        newdata = data.discretize_variables(self.data, numbins=3, excludevars=[0,1])
        assert newdata.observations[:,1].tolist() == self.data.observations[:,1].tolist(), "Don't discretize variable if in excludevars."
        assert newdata.observations[:,2].tolist() == [1, 1, 0, 2, 0, 1, 2, 0, 2], "Discretize variable if not in excludevars."


class TestDataIncorrectArity:
    def test_uniquevals_morethan_arity(self):
        try:
            dat = data.fromfile(os.path.join(datadir(), 'testdata6.txt'))
            assert False, "Should have raised an exception"
        except data.IncorrectArityError:
            assert True
            
    def test_uniquevals_lessthan_arity(self):
        try:
            dat = data.fromfile(os.path.join(datadir(), 'testdata7.txt'))
        except data.IncorrectArityError:
            assert False, "Shouldn't raise error when arity is MORE than the number of unique observations."



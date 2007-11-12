"""A Class for representing datasets and functions to create and manipulate them."""

import re
import numpy as N
from itertools import groupby
from util import *

class ParsingError(Exception): pass

class Annotation(Struct):
    def __repr__(self):
        name = getattr(self, 'name', self.id)
        return "<%s: %s>" % (self.__class__.__name__,  name)

class Variable(Annotation): pass
class Sample(Annotation): pass

class PeblData(object):
    def __init__(self, observations, missing, interventions, variables, samples):
        self.observations = observations
        self.missing = missing
        self.interventions = interventions
        self.variables = variables
        self.samples = samples

        if not hasattr(self.variables[0], 'arity'):
            self._guess_arities()

    def subset(self, variables=None, samples=None):
        """Return a subset of the dataset."""

        variables = variables if variables is not None else range(self.variables.size)
        samples = samples if samples is not None else range(self.samples.size)

        return PeblData(
            self.observations[N.ix_(samples,variables)],
            self.missing[N.ix_(samples,variables)],
            self.interventions[N.ix_(samples,variables)],
            self.variables[variables],
            self.samples[samples]
        )

    @property
    def shape(self):
        return self.observations.shape

    @property
    def arities(self):
        return [v.arity for v in self.variables]

    def _1d_where(self, arr, predicate):
        return [i for i,a in enumerate(arr) if predicate(a)]

    def variables_where(self, predicate):
        return self._1d_where(self.variables, predicate)

    def samples_where(self, predicate):
        return self._1d_where(self.samples, predicate)

    def _guess_arities(self):
        """Tries to guess arity of variables by counting the number of unique observations."""

        for var in self.variables:
            var.arity = N.unique(self.observations[:,var]).size

    def _NOT_READY_tofile(self, filename):
        """Write the data and metadata to file in a tab-delimited format."""
        
        f = file(filename, 'w')
        f.write(self.tostring())
        f.close()
    
    def _NOT_READY_tostring(self, linesep='\n'):
        """Return the data and metadata as a String."""

        def format_item(row, col):
            val = "X" if self.missing[row,col] else self.observations[row,col]
            val += "!" if self.interventions[row,col] else ''
            return val
        
        lines = []
        # STEPS:
        #   * determine annotations by inspecting variables.dtype, samples.dtype
        #   * add control lines based on the annotations
        #   * ass data using format_item

        return linesep.join(lines)

example_data = """
%variable.name[str]=foo\tbar\tfoobar\tbaz\tfoobaz
%variable.arity[int]=2\t3\t3\t2\t3
%sample.adult_sample[bool]=0\t1\t1\t0
0\t1\t0\t1\t0
1\t0\t2\tx\t0!
2\t1\t0!\t1!\tx
!0\tX\tx\t0\t1
"""
     
def fromstring(stringrep):
    # parse a data item (examples: '5' '2.5', 'X', '5!')
    def parse_dataitem(item):
        item = item.strip()

        intervention = False
        missing = False
        if item[0] == "!":
            intervention = True
            item = item[1:]
        elif item[-1] == "!":
            intervention = True
            item = item[:-1]

        if item[0] in ('x', 'X') or item[-1] in ('x', 'X'):
            missing = True
            item = "0"

        # first try converting to int, then to float
        try:
            value = int(item)
        except ValueError:
            try:
                value = float(item)
            except:
                raise ValueError("Error parsing value: '%s'. It doesn't seem to be an int or a float." % item)

        return (value, missing, intervention)

    metadata_re = re.compile("([^\.]*\.)?(.*)\[(.*)\]=(.*)$")
    bool_converter = lambda x: bool(int(x))
    # parse metadata line
    def parse_metadata(line):
        """Parses metadata annotations like '%variable.arity[int]=2\t3\t2\t2'"""
        
        match = metadata_re.match(line[1:])  # line[1:] to remove the %
        if not match:
            raise ParsingError("Error parsing metadata line: %s" % line)

        annotfor, name, datatype, values = match.groups()
        annotfor = annotfor[:-1]  # remove the trailing period from annotfor
        
        try:
            converter = bool_converter if datatype == 'bool' else eval(datatype) 
        except NameError:
            raise ParsingError(
                "Invalid datatype (%s) for annotation (%s.%s)." % (datatype, annotfor, name)
            )
   
        values = [converter(v) for v in re.split(',|\t', values)]
        return (annotfor, name, values)
    
    # create array of variable or sample annotations
    def annotations(annotation_type, metadata, length):
        names = [m[1] for m in metadata]
        values = [m[2] for m in metadata]

        if 'id' not in names:
            names.append('id')
            values.append(range(length))

        valuesets = unzip(values)  

        return array([annotation_type(**dict((n,v) for n,v in zip(names, values))) for values in valuesets])

    # -------------------------------------------------------------------------------------------------

    # split on all known line seperators, then ignore blank and comment lines
    lines = (l.strip() for l in stringrep.splitlines() if l)
    lines = (l for l in lines if not l.startswith('#'))
    
    # separate and then parse metadata and data lines 
    linetype = lambda line: 'metadata' if line.startswith('%') else 'data'
    for group,grouplines in groupby(lines, linetype):
        if group is 'metadata':
            metadata = [parse_metadata(l) for l in grouplines]
        else:
            data_ = N.array([[parse_dataitem(i) for i in l.split('\t')] for l in grouplines])
            # data_ is a 3D array where the inner dimension is over (values, missing, interventions)
            # transpose(2,0,1) makes the inner dimension the outer one
            observations, missing, interventions = data_.transpose(2,0,1)

    # create variable and sample annotations from the metadata
    variables = annotations(Variable, [m for m in metadata if m[0] == 'variable'], observations.shape[1])
    samples = annotations(Sample, [m for m in metadata if m[0] == 'sample'], observations.shape[0])

    # pack observations into bytes if possible (they're integers and < 255)
    observations_dtype = 'byte' if (observations.dtype.kind is 'i' and observations.max() < 255) \
                                else observations.dtype
   
    # x.astype() returns a casted *copy* of x
    # returning copies of observations, missing and interventions ensures that
    # they're contiguous in memory (should speedup future calculations)
    return PeblData(
        observations.astype(observations_dtype),
        missing.astype(bool),
        interventions.astype(bool), 
        variables, 
        samples
    )


# This implementation calculates bin edges by trying to make bins equal sized.. 
# 
# input:  [3,7,4,4,4,5]
# output: [0,1,0,0,0,1]
#
# Note: All 4s discretize to 0.. which makes bin sizes unequal..                                                 
def discretize_variables(data_, includevars=None, excludevars=[], numbins=3):
    newdata = copy.deepcopy(data_)
    includevars = ensure_list(includevars or range(len(newdata.variables)))
    binsize = len(newdata.samples)//numbins

    vars = (var for var in includevars if var not in excludevars)
    for var in vars:
        row = newdata.observations[:,var]
        argsorted = row.argsort()
        binedges = [row[argsorted[binsize*b - 1]] for b in range(numbins)][1:]
        newdata.observations[:,var] = N.searchsorted(binedges, row)
        newdata.variables[var].arity = numbins

    return newdata

def discretize_variables_in_groups(data_, samplegroups, includevars=None, excludevars=[], numbins=3):
    newdata = copy.deepcopy(data_)

    includevars = ensure_list(includevars or range(len(newdata.variables)))
    
    for samplegroup in samplegroups:
        newdata.observations[samplegroup] = newdata.discretize_variables(
                                                newdata.observations[samplegroup], 
                                                includevars, excludevars, numbins)

    return newdata


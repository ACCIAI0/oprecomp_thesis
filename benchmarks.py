#!usr/bin/python

import os
import glob

import vargraph

benchmarks_home = 'flexfloat-benchmarks/'


class Benchmark:

    def __init__(self, name):
        self.name = name
        self.graph = vargraph.parse_vars_file(benchmarks_home + name + '/program_vardeps.json')
        self.nVars = len(self.graph)
        print("{}\n{}".format(self.nVars, self.graph))

    def plot_var_graph(self):
        vargraph.plot(self.graph)


def exists(name):
    return name in __available.keys()


def get_benchmark(name):
    return __available[name]


__available = {el.name: el for el in [Benchmark(os.path.basename(x)) for x in glob.glob(benchmarks_home + '*')]}

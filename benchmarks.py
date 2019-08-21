#!usr/bin/python

import os
import glob
import json

import vargraph

benchmarks_home = 'flexfloat-benchmarks/'


class Benchmark:

    def __init__(self, name):
        self.__name = name
        self.__lazyEval = False
        self.__graph = None
        self.__nVars = -1
        self.__home = benchmarks_home + name + "/"
        self.__configs = self.__home + "config_file.txt"

        jdict = {}
        with open(self.__home + "global_info.json") as jfile:
            jdict = json.load(jfile)
        self.__map = jdict.get('map', None)
        self.__flag = jdict.get('flag', False)

    def __evaluate(self):
        if not self.__lazyEval:
            self.__graph = vargraph.parse_vars_file(self.__home + "program_vardeps.json")
            self.__nVars = len(self.__graph)
            self.__lazyEval = True

    def get__name(self) -> str:
        return self.__name

    def get_graph(self):
        self.__evaluate()
        return self.__graph

    def get_vars_number(self) -> int:
        self.__evaluate()
        return self.__nVars

    def get_home(self):
        return self.__home

    def get_map(self):
        return self.__map

    def get_configs_file(self):
        return self.__configs

    def is_flagged(self):
        return self.__flag

    def plot_var_graph(self):
        vargraph.plot(self.get_graph())

    def get_binary_relations(self) -> dict:
        return {
            'leq': vargraph.extract_leq_relations(self.get_graph()),
            'cast': vargraph.extract_cast_to_temp_relations(self.get_graph())
        }


def exists(name: str) -> bool:
    """
    Checks if a benchmark with the given name exists.
    :param name: the name of the benchmark to check the existence of.
    :return: True if the benchmark with the given name exists, False otherwise.
    """
    return name in __available.keys()


def get_benchmark(name: str) -> Benchmark:
    """
    Returns a Benchmark object having the given name.
    :param name: The name of the Benchmark.
    :return: a Benchmark object relative to the given benchmark name, None if it doesn't exist.
    """
    return __available.get(name, None)


__available = {el.get__name(): el for el in [Benchmark(os.path.basename(x)) for x in glob.glob(benchmarks_home + '*')]}

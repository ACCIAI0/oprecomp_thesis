#!usr/bin/python

import subprocess
import os
import glob
import json

import vargraph
from utils import io_utils

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


def __check_output(floating_result, target_result):
    very_large_error = 1000
    if len(floating_result) != len(target_result) or \
            all(v == 0 for v in floating_result) != all(v == 0 for v in target_result):
        return very_large_error

    sqnr = 0.00
    for i in range(len(floating_result)):
        # if floating_result[i] == 0, check_output returns 1: this is an
        # unwanted behaviour
        if floating_result[i] == 0.00:
            continue    # mmmhhh, TODO: fix this in a smarter way

        # if there is even just one inf in the result list, we assume that
        # for the given configuration the program did not run properly
        if str(floating_result[i]) == 'inf':
            return float('nan')

        signal_sqr = target_result[i] ** 2
        error_sqr = (floating_result[i] - target_result[i]) ** 2
        temp = 0.00
        if error_sqr != 0.00:
            temp = signal_sqr / error_sqr
            temp = 1.0 / temp
        if temp > sqnr:
            sqnr = temp

    return sqnr


def __run_check(program, target_result, dataset_index):
    params = [program, '42']
    if -1 != dataset_index:
        params.append(str(dataset_index))
    output = subprocess.Popen(params, stdout=subprocess.PIPE).communicate()[0]
    result = io_utils.parse_output(output.decode('utf-8'))
    return __check_output(result, target_result)


def run_benchmark_with_config(bm: Benchmark, opt_config, args):
    """
    Runs a Benchmark with a given bit numbers configuration and returns the actual error such a configuration creates.
    :param bm: the benchmark to run.
    :param opt_config: the configuration of bit numbers to use in the run.
    :return: the actual error given the configuration.
    """

    io_utils.write_configs_file(bm.get_configs_file(), opt_config)
    if -1 == args.get_dataset_index():
        program = bm.get_home() + bm.get_map() + '.sh'
        target_file = bm.get_home() + 'target.txt'
    else:
        program = bm.get_home() + bm.get_map() + '_multiDataSet.sh'
        target_file = bm.get_home() + 'targets/target_{}.txt'.format(args.get_dataset_index())

    target_result = io_utils.read_target(target_file)
    error = __run_check(program, target_result, args.get_dataset_index())

    io_utils.write_configs_file(bm.get_configs_file(), [args.get_max_bits_number()] * bm.get_vars_number())
    return error


__available = {el.get__name(): el for el in [Benchmark(os.path.basename(x)) for x in glob.glob(benchmarks_home + '*')]}

#!/usr/bin/python

import math

import pandas

import benchmarks
import argsmanaging

from eml.backend import cplex_backend


def create_optimization_model(args: argsmanaging.ArgumentsHolder, benchmark: benchmarks.Benchmark,
                              regressor, classifier, robustness: int = 0):
    target_error_log = math.ceil(args.error)
    max_config = pandas.DataFrame.from_dict({'var_{}'.format(i): [args.maxBitsNumber]
                                             for i in range(benchmark.get_vars_number())})
    max_predictable_error = regressor.predict(max_config)[0][0]

    if max_predictable_error < args.error:
        robustness = -math.ceil(args.error - max_predictable_error)

    backend = cplex_backend.CplexBackend()

    model = # TODO

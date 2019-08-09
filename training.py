#!/usr/bin/python

import math

import numpy
import pandas

import argsmanaging as argsm
import benchmarks as bm

__dataset_home = "datasets/"

__error_0_threshold = .05
__error_high_threshold = .9
__clamped_target_error = -numpy.log(80)


def __prepare_dataframe(dataset_size: int, benchmark: bm.Benchmark, clamp=True):
    data_file = 'exp_results_{}_{}.csv'.format(benchmark.get__name(), dataset_size)

    dataframe = pandas.read_csv(__dataset_home + data_file)


def create_training_test(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, set_size: int = 500):
    pass
    # TODO Prepare training set <- returns a Pandas dataframe
    # TODO Filters the dataframe and remove useless columns
    # TODO Artificially set configs with no error (AKA error = 0) to "large error logs" (?)
    # TODO Merge all data
    # TODO Generate class attribute
    # TODO TODO rest

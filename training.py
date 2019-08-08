#!/usr/bin/python

import argsmanaging as argsm
import benchmarks as bm

__dataset_home = "datasets/"


def create_training_test(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, set_size: int = 500):
    pass
    # TODO Prepare training set <- returns a Pandas dataframe
    # TODO Filters the dataframe and remove useless columns
    # TODO Artificially set configs with no error (AKA error = 0) to "large error logs" (?)
    # TODO Merge all data
    # TODO Generate class attribute
    # TODO TODO rest

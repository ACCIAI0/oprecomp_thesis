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
    data_file = 'exp_results_{}.csv'.format(benchmark.get__name())

    dataframe = pandas.read_csv(__dataset_home + data_file, sep=';')
    del dataframe['err_mean']
    del dataframe['err_std']

    # Clamp high errors to a reasonable value
    if clamp:
        for col in dataframe.columns:
            if 'err_ds_' in col:
                dataframe.loc[dataframe[col] > __error_high_threshold, col] = __error_high_threshold

    logs = []
    errs = []
    for col in dataframe.columns:
        if 'err_' in col:
            name = 'log_{}'.format(col)
            dataframe[name] = -numpy.log(dataframe[col])
            if 'std' not in name and 'mean' not in name:
                logs.append(name)
                errs.append(col)
    dataframe['log_err_mean'] = numpy.mean(dataframe[logs], axis=1)
    dataframe['log_err_std'] = numpy.std(dataframe[logs], axis=1)
    dataframe['err_mean'] = numpy.mean(dataframe[errs], axis=1)
    dataframe['err_std'] = numpy.std(dataframe[errs], axis=1)

    # Clamp to a bit over 0 errors, since -log(0) = infinite
    dataframe.loc[dataframe['err_std'] == 0, 'err_std'] = __error_0_threshold

    if dataset_size < len(dataframe):
        dataframe = dataframe.sample(dataset_size)
    return dataframe


def __filter_and_fill(df: pandas.DataFrame, ds_index: int):
    dataset_label = 'ds_{}'.format(ds_index)
    error_label = 'err_' + dataset_label
    log_error_label = 'log_err_' + dataset_label
    class_error_label = 'err_class_' + dataset_label

    deletion = []
    for col in df.columns:
        if 'var_' not in col and dataset_label not in col:
            deletion.append(col)
    for col in deletion:
        del df[col]
    df_error_zero = df[df[error_label] == 0]
    df = df[df[error_label] != 0]
    df_error_zero[log_error_label] = math.inf
    frames = [df, df_error_zero]
    df = pandas.concat(frames)[(df != 0).all(1)]
    df[class_error_label] = df.apply(lambda c: int(c[error_label] >= __error_high_threshold), axis=1)
    return df.drop_duplicates()


def create_training_test(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, set_size: int = 500):
    # Prepare training set <- returns a Pandas dataframe
    df = __prepare_dataframe(set_size, benchmark)

    # Filters the set and removes all columns from other dataset with a different index. This also artificially sets
    # all log errors of configurations with 'err_ds_{index}' = 0 to infinite (n / 0 would be an error). Finally, it adds
    # an additional label attribute for the classifier.
    df = __filter_and_fill(df, args.datasetIndex)

    return df

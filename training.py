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


def __prepare_dataframe_lhs(dataset_size: int, benchmark: bm.Benchmark, clamp=True):
    data_file = 'exp_results_{}.csv'.format(benchmark.get__name())

    dataframe = pandas.read_csv(__dataset_home + data_file, sep=';')
    del dataframe['err_mean']
    del dataframe['err_std']

    # Clamp high errors to a reasonable value
    if clamp:
        for col in dataframe.columns:
            if 'err_ds_' in col:
                dataframe.loc[dataframe[col] > __error_high_threshold, col] = __error_high_threshold

    if dataset_size < len(dataframe):
        dataframe = dataframe.sample(dataset_size)
    return dataframe


def __filter(df: pandas.DataFrame, dataset_label, error_label, log_error_label):
    deletion = []
    for col in df.columns:
        if 'var_' not in col and dataset_label not in col:
            deletion.append(col)
    for col in deletion:
        del df[col]
    df.loc[df[error_label] == 0] = math.inf
    df[log_error_label] = -numpy.log(df[error_label])
    return df[(df != 0).all(1)]


def __calculate_class(df: pandas.DataFrame, error_label, class_error_label):
    df[class_error_label] = df.apply(lambda c: int(c[error_label] >= __error_high_threshold), axis=1)


def __select_subset(df: pandas.DataFrame, size: int, error_label: str):
    n_large_errors = len(df[df[error_label] >= __error_high_threshold])
    ratio = n_large_errors / len(df)

    if 0 == ratio:
        return df.sample(size)

    acc = 0
    while acc < ratio:
        if size > len(df):
            size = len(df) - 1
        df_t = df.sample(size)
        acc = len(df_t[error_label] >= __error_high_threshold) / len(df_t)
    return df_t


def create_training_test(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, set_size: int = 500):
    dataset_label = 'ds_{}'.format(args.datasetIndex)
    error_label = 'err_' + dataset_label
    log_error_label = 'log_err_' + dataset_label
    class_error_label = 'class_' + dataset_label
    
    # Prepare training set <- returns a Pandas dataframe
    df = __prepare_dataframe_lhs(set_size, benchmark)

    # Filters the set and removes all columns from other dataset with a different index. This also artificially sets
    # all log errors of configurations with 'err_ds_{index}' = 0 to infinite (n / 0 would be an error).
    df = __filter(df, dataset_label, error_label, log_error_label)

    # Finally, it adds an additional label attribute for the classifier.
    __calculate_class(df, error_label, class_error_label)

    target_regressor = df[log_error_label]
    target_classifier = df[class_error_label]
    del df[error_label]
    del df[log_error_label]
    del df[class_error_label]
    return df.drop_duplicates(), target_regressor, target_classifier


def create_test_set(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, set_size: int = 9000):
    dataset_label = 'ds_{}'.format(args.datasetIndex)
    error_label = 'err_' + dataset_label
    log_error_label = 'log_err_' + dataset_label
    class_error_label = 'class_' + dataset_label

    # Prepare training set <- returns a Pandas dataframe
    df = __prepare_dataframe_lhs(set_size, benchmark)

    df = df.sample(frac=1).reset_index(drop=True)
    df = __select_subset(df, set_size, error_label)

    # Filters the set and removes all columns from other dataset with a different index and. This also artificially sets
    # all log errors of configurations with 'err_ds_{index}' = 0 to infinite (n / 0 would be an error).
    df = __filter(df, dataset_label, error_label, log_error_label)

    # Finally, it adds an additional label attribute for the classifier.
    __calculate_class(df, error_label, class_error_label)

    train_regressor = df[log_error_label]
    train_classifier = df[class_error_label]
    del df[error_label]
    del df[log_error_label]
    del df[class_error_label]
    return df, train_regressor, train_classifier

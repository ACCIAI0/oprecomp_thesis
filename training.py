#!/usr/bin/python

import math

import numpy
import pandas

import argsmanaging as argsm
import benchmarks as bm

__dataset_home = "datasets/"

__error_0_threshold = .05
__error_high_threshold = .9
__clamped_error_limit = .9
__clamped_target_error = -numpy.log(80)


class TrainingSession:
    def __init__(self, training_set, test_set, regressor_target_label, classifier_target_label):
        self.__targetRegressor = training_set[regressor_target_label]
        self.__targetClassifier = training_set[classifier_target_label]

        del training_set[regressor_target_label]
        del training_set[classifier_target_label]

        self.__trainingSet = training_set

        self.__testRegressor = test_set[regressor_target_label]
        self.__testClassifier = test_set[classifier_target_label]

        del test_set[regressor_target_label]
        del test_set[classifier_target_label]

        self.__testSet = test_set

    def get_training_set(self):
        return self.__trainingSet

    def get_test_set(self):
        return self.__testSet

    def get_target_regressor(self):
        return self.__targetRegressor

    def get_target_classifier(self):
        return self.__targetClassifier

    def get_test_regressor(self):
        return self.__testRegressor

    def get_test_classifier(self):
        return self.__testClassifier


def __initialize_mean_std(benchmark: bm.Benchmark, index: int, clamp: bool = True):
    data_file = 'exp_results_{}.csv'.format(benchmark.get__name())
    label = 'err_ds_{}'.format(index)
    log_label = 'err_log_ds_{}'.format(index)
    class_label = 'class_ds_{}'.format(index)

    df = pandas.read_csv(__dataset_home + data_file, sep=';')

    columns = [x for x in filter(lambda l: 'var_' in l or label == l, df.columns)]
    df = df[columns]

    if clamp:
        df.loc[df[label] > __clamped_error_limit, label] = __clamped_error_limit
    df[log_label] = [math.inf if 0 == x else -numpy.log(x) for x in df[label]]
    df[class_label] = df.apply(lambda e: int(e[label] >= __error_high_threshold), axis=1)
    return df


def __select_subset(df: pandas.DataFrame, size: int, index: int):
    error_label = 'err_ds_{}'.format(index)
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


def create_training_session(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark,
                            initial_sampling_size: int = 3000, set_size: int = 500):
    df = __initialize_mean_std(benchmark, args.datasetIndex)
    df = __select_subset(df, initial_sampling_size, args.datasetIndex)
    df = df.reset_index(drop=True)
    df = df[(df != 0).all(1)]
    del df['err_ds_{}'.format(args.datasetIndex)]
    return TrainingSession(df[:set_size], df[set_size:],
                           'err_log_ds_{}'.format(args.datasetIndex), 'class_ds_0'.format(args.datasetIndex))

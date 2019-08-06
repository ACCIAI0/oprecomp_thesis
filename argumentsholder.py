#!/usr/bin/python

from enum import Enum

import numpy

import benchmarks


class ArgsError(Enum):
    NO_ERROR = 0
    UNKNOWN_BENCHMARK = 1
    INT_CAST_ERROR = 2
    REGRESSOR_ERROR = 3
    CLASSIFIER_ERROR = 4


class Regressor(Enum):
    NEURAL_NETWORK = 'NN'


class Classifier(Enum):
    DECISION_TREE = 'DT'


class Init:

    def __init__(self, benchmark=None, error=-1.0, regressor=Regressor.NEURAL_NETWORK,
                 classifier=Classifier.DECISION_TREE, dataset_index=0):
        self.benchmark = benchmark
        self.error = error
        self.regressor = regressor
        self.classifier = classifier
        self.datasetIndex = dataset_index

    def is_legal(self):
        return self.benchmark is not None and self.error is not -1


def int_value(value):
    error = ArgsError.NO_ERROR
    v = 0
    try:
        v = int(value)
    except ValueError:
        error = ArgsError.INT_CAST_ERROR
    return error, v


def __benchmark(init, value):
    error = ArgsError.NO_ERROR
    if not benchmarks.exists(value):
        error = ArgsError.UNKNOWN_BENCHMARK
    if ArgsError.NO_ERROR == error:
        init.benchmark = value
    return error, value


def _exp(init, value):
    error, v = int_value(value)
    if ArgsError.NO_ERROR == error:
        init.error = -numpy.log(numpy.power(1.0, -v))
    return error, v


def __regressor(init, value):
    error = ArgsError.NO_ERROR
    if value not in [reg.value for reg in Regressor]:
        error = ArgsError.REGRESSOR_ERROR
    if ArgsError.NO_ERROR == error:
        init.regressor = Regressor(value)
    return error, value


def __classifier(init, value):
    error = ArgsError.NO_ERROR
    if value not in [cl.value for cl in Classifier]:
        error = ArgsError.CLASSIFIER_ERROR
    if ArgsError.NO_ERROR == error:
        init.classifier = Classifier(value)
    return error, value


def __dataset(init, value):
    error, v = int_value(value)
    if ArgsError.NO_ERROR == error:
        init.datasetIndex = v
    return error, v


def error_handler(e, param, value):
    assert isinstance(param, str)
    param = param.replace('-', '')
    if ArgsError.UNKNOWN_BENCHMARK == e:
        print("Can't find a benchmark called {}".format(value))
    elif ArgsError.INT_CAST_ERROR == e:
        print("Expected an integer value, found '{}' as {}".format(value, param))
    elif ArgsError.REGRESSOR_ERROR == e:
        s = ''
        for reg in Regressor:
            s += reg.value + ', '
        s = s[:len(s) - 2]
        print("Invalid regressor {}. Possible values are: {}".format(value, s))
    elif ArgsError.CLASSIFIER_ERROR == e:
        s = ''
        for cl in Classifier:
            s += cl.value + ', '
        s = s[:len(s) - 2]
        print("Invalid classifier {}. Possible values are: {}".format(value, s))

    if ArgsError.NO_ERROR is not e:
        exit(e.value)


__args = {
    '-bm': __benchmark,
    '-exp': _exp,
    '-r': __regressor,
    '-c': __classifier,
    '-dataset': __dataset
}


def handle_args(argv):
    init = Init()
    if argv[0] == '-help':
        s = ''
        for a in __args.keys():
            s += a + ', '
        s = s[:len(s) - 2]
        print("Possible parameters: {}".format(s))
        exit(0)

    for i in range(0, len(argv) - 1, 2):
        p = argv[i]
        v = None

        if not p.startswith('-'):
            print("Invalid parameter name {}".format(p))
            exit(1)

        if i + 1 < len(argv):
            v = argv[i + 1]
        error, value = __args[p](init, v)
        error_handler(error, p, v)
    if not init.is_legal():
        print("Benchmark and error exponent are mandatory")
        exit(1)
    return init

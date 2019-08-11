#!/usr/bin/python

from enum import Enum

import numpy

import benchmarks


class ArgsError(Enum):
    NO_ERROR = 0
    UNKNOWN_BENCHMARK = 1
    INT_CAST_ERROR = 2
    INT_INVALID_VALUE = 3
    REGRESSOR_ERROR = 4
    CLASSIFIER_ERROR = 5


class Regressor(Enum):
    NEURAL_NETWORK = 'NN'


class Classifier(Enum):
    DECISION_TREE = 'DT'


class ArgumentsHolder:

    def __init__(self, benchmark: benchmarks.Benchmark = None, error: float = -1.0,
                 regressor: Regressor = Regressor.NEURAL_NETWORK, classifier: Classifier = Classifier.DECISION_TREE,
                 dataset_index: int = 0, min_bits: int = 4, max_bits: int = 53):
        self.benchmark = benchmark
        self.error = error
        self.regressor = regressor
        self.classifier = classifier
        self.datasetIndex = dataset_index
        self.minBitsNumber = min_bits
        self.maxBitsNumber = max_bits

    def is_legal(self):
        return self.benchmark is not None and self.error is not -1.0

    def __str__(self):
        return "BENCHMARK:\t\t\t{}\nTARGET ERROR:\t\t{}\nREGRESSOR:\t\t\t{}\nCLASSIFIER:\t\t\t{}\n" \
               "DATASET #:\t\t\t{}\nBITS NUMBER:\t\t[{}, {}]"\
            .format(self.benchmark, self.error, self.regressor, self.classifier,
                    self.datasetIndex, self.minBitsNumber, self.maxBitsNumber)


def __int_value(value):
    error = ArgsError.NO_ERROR
    v = 0
    try:
        v = int(value)
        if 0 >= v:
            error = ArgsError.INT_INVALID_VALUE
    except ValueError:
        error = ArgsError.INT_CAST_ERROR
    return error, v


def __benchmark(args, value):
    error = ArgsError.NO_ERROR
    if not benchmarks.exists(value):
        error = ArgsError.UNKNOWN_BENCHMARK
    if ArgsError.NO_ERROR == error:
        args.benchmark = value
    return error, value


def _exp(args, value):
    error, v = __int_value(value)
    if ArgsError.NO_ERROR == error:
        args.error = -numpy.log(numpy.float_power(10, -v))
    return error, args.error


def __regressor(args, value):
    error = ArgsError.NO_ERROR
    if value not in [reg.value for reg in Regressor]:
        error = ArgsError.REGRESSOR_ERROR
    if ArgsError.NO_ERROR == error:
        args.regressor = Regressor(value)
    return error, value


def __classifier(args, value):
    error = ArgsError.NO_ERROR
    if value not in [cl.value for cl in Classifier]:
        error = ArgsError.CLASSIFIER_ERROR
    if ArgsError.NO_ERROR == error:
        args.classifier = Classifier(value)
    return error, value


def __dataset(args, value):
    error, v = __int_value(value)
    if ArgsError.NO_ERROR == error:
        args.datasetIndex = v
    return error, v


def __min_bits(args, value):
    error, v = __int_value(value)
    if ArgsError.NO_ERROR == error:
        args.minBitsNumber = v
    return error, v


def __max_bits(args, value):
    error, v = __int_value(value)
    if ArgsError.NO_ERROR == error:
        args.maxBitsNumber = v
    return error, v


def error_handler(e, param, value):
    assert isinstance(param, str)
    param = param.replace('-', '')
    if ArgsError.UNKNOWN_BENCHMARK == e:
        print("Can't find a benchmark called {}".format(value))
    elif ArgsError.INT_CAST_ERROR == e:
        print("Expected an integer value, found '{}' as {}".format(value, param))
    elif ArgsError.INT_INVALID_VALUE == e:
        print("{} value must be greater than 0".format(param))
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
    '-dataset': __dataset,
    '-minb': __min_bits,
    '-maxb': __max_bits
}


def handle_args(argv):
    """
        Handles the arguments passed as parameters at the start of the program. It can cause the entire program to quit
        if any error is encountered (see ArgsError for the possible error types).
        :param argv: arguments array. from the starting script, it is the system arguments list slice from the second
        argument onward.
        :return: an ArgumentHolder if all arguments are legal. If any of them is not, the program quits.
    """

    args = ArgumentsHolder()

    if 0 == len(argv):
        print("Parameters needed. Use -help for all possibilities.")
        exit(-1)

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
        error, value = __args[p](args, v)
        error_handler(error, p, v)
    if not args.is_legal():
        print("Benchmark and error exponent are mandatory")
        exit(1)
    return args

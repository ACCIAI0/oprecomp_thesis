#!/usr/bin/python

import sys
from enum import Enum
import numpy

from init import Init, Regressor, Classifier
import benchmarks


class ArgsError(Enum):
	NO_ERROR = 0
	UNKNOWN_BENCHMARK = 1
	INT_CAST_ERROR = 2
	REGRESSOR_ERROR = 3
	CLASSIFIER_ERROR = 4


def int_value(value):
	error = ArgsError.NO_ERROR
	v = 0
	try:
		v = int(value)
	except ValueError:
		error = ArgsError.INT_CAST_ERROR
	return error, v


def benchmark(init, value):
	error = ArgsError.NO_ERROR
	if not benchmarks.exists(value):
		error = ArgsError.UNKNOWN_BENCHMARK
	if ArgsError.NO_ERROR == error:
		init.benchmark = value
	return error, value


def exp(init, value):
	error, v = int_value(value)
	if ArgsError.NO_ERROR == error:
		init.error = -numpy.log(numpy.power(1.0, -v))
	return error, v


def r(init, value):
	error = ArgsError.NO_ERROR
	if value not in [reg.value for reg in Regressor]:
		error = ArgsError.REGRESSOR_ERROR
	if ArgsError.NO_ERROR == error:
		init.regressor = Regressor(value)
	return error, value


def c(init, value):
	error = ArgsError.NO_ERROR
	if value not in [cl.value for cl in Classifier]:
		error = ArgsError.CLASSIFIER_ERROR
	if ArgsError.NO_ERROR == error:
		init.classifier = Classifier(value)
	return error, value


def dataset(init, value):
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


args = {
	'-bm'		: benchmark,
	'-exp'		: exp,
	'-r'		: r,
	'-c'		: c,
	'-dataset'	: dataset
}


def handle_args(argv):
	init = Init()
	if argv[0] == '-help':
		s = ''
		for a in args.keys():
			s += a + ', '
		s = s[:len(s)-2]
		print("Possible parameters: {}".format(s))
		exit(0)

	for i in range(0, len(argv) - 1, 2):
		p = argv[i]
		v = None

		if not p.startswith('-'):
			print("Invalid parameter name {}".format(p))
			exit(1)

		if i + 1 < len(argv):
			v = argv[i+1]
		error, value = args[p](init, v)
		error_handler(error, p, v)
	if not init.is_legal():
		print("Benchmark and error exponent are mandatory")
		exit(1)
	return init


def main(argv):
	init = handle_args(argv)
	benchmarks.get_benchmark(init.benchmark).plot_var_graph()
	# TODO Add relations from graph??
	# TODO Get training set data
	# TODO Create regressor 	<- Already trained at the end
	# TODO Create classifier 	<- Ditto
	# TODO Create a MP model
	# TODO solve optimization problem
	# TODO FINAL CHECK BEING... who knows





'''
Entry point. Call to main function if this is main module.
'''
if __name__ == '__main__':
	main(sys.argv[1:])

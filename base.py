#!/usr/bin/python

import sys

import argsmanaging as am
import benchmarks
import training


def main(argv):
	# ArgumentsHolder containing all legal initialization params.
	args = am.handle_args(argv)

	# Benchmarks information. Contains a relational graph among variables inside the benchmark program and the number
	# of them.
	bm = benchmarks.get_benchmark(args.benchmark)

	# Get training set data
	training_set = training.create_training_test(args, bm)
	print(training_set)
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

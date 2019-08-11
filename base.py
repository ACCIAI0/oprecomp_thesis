#!/usr/bin/python

import sys

from stopwatch import Stopwatch

import argsmanaging as am
import benchmarks
import training

DEBUG = True


def main(argv):
	stop_w = Stopwatch()

	# ArgumentsHolder containing all legal initialization params.
	args = am.handle_args(argv)
	if DEBUG:
		print(args)
		print("============================================\n")

	# Benchmarks information. Contains a relational graph among variables inside the benchmark program and the number
	# of them.
	stop_w.start()
	bm = benchmarks.get_benchmark(args.benchmark)
	_, t = stop_w.stop()
	print("{} loaded in {}s ({} variables)".format(bm.get__name(), t, bm.get_vars_number()))

	# Get training set data
	stop_w.start()
	training_set, target_regressor, target_classifier = training.create_training_test(args, bm, 900)
	_, t = stop_w.stop()
	print("Created training set from dataset #{} in {}s ({} entries)".format(args.datasetIndex, t, len(training_set)))

	# Get test set data
	stop_w.start()
	test_set, test_regressor, test_classifier = training.create_test_set(args, bm)
	_, t = stop_w.stop()
	print("Created test set from dataset #{} in {}s ({} entries)".format(args.datasetIndex, t, len(test_set)))

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

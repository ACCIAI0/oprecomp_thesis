#!/usr/bin/python

import sys

import argsmanaging as am
import benchmarks


def main(argv):
	# ArgumentsHolder containing all legal initialization params.
	args = am.handle_args(argv)

	# Benchmarks information. Contains a relational graph among variables inside the benchmark program and the number
	# of them.
	bm = benchmarks.get_benchmark(args.benchmark)
	print(bm.get_vars_number())
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

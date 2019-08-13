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
    bm.get_graph()
    _, t = stop_w.stop()
    print("{} loaded in {}s ({} variables)".format(bm.get__name(), t, bm.get_vars_number()))

    # Get training set data
    stop_w.start()
    session = training.create_training_session(args, bm, set_size=900)
    _, t = stop_w.stop()
    print("Created training session from dataset #{} in {}s ({} entries from training, {} for test)"
          .format(args.datasetIndex, t, len(session.get_training_set()), len(session.get_test_set())))


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

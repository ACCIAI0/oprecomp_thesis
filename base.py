#!/usr/bin/python

import warnings
warnings.filterwarnings('ignore')

import sys
import os

from stopwatch import Stopwatch
import argsmanaging as am
import benchmarks
import training

import eml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    print("{} loaded in {:.3f}s ({:d} variables)".format(bm.get__name(), t, bm.get_vars_number()))

    # Build training set and test set for a new training session
    stop_w.start()
    session = training.create_training_session(args, bm, set_size=900)
    _, t = stop_w.stop()
    print("Created first training session from dataset #{:d} in {:.3f}s ({:d} entries for training, {:d} for test)"
          .format(args.datasetIndex, t, len(session.get_training_set()), len(session.get_test_set())))

    # Train a regressor
    stop_w.start()
    model, r_stats = training.regressor_trainings[args.regressor](args, bm, session)
    _, t = stop_w.stop()
    print("Trained the regressor in {:.3f}s (MAE {:.3f})".format(t, r_stats['MAE']))

    # Train a classifier
    stop_w.start()
    classifier, c_stats = training.classifier_trainings[args.classifier](args, bm, session)
    _, t = stop_w.stop()
    print("Trained the classifier in {:.3f}s (accuracy {:.3f}%)".format(t, c_stats['accuracy'] * 100))

    # TODO Create a MP model
    # TODO solve optimization problem
    # TODO FINAL CHECK BEING... who knows

    print("\n+-- TOTAL EXECUTION TIME: {:.3f}s --+".format(stop_w.get_duration()))


'''
Entry point. Imports EML and calls to main function if this is main module.
'''
if __name__ == '__main__':
    # eml_path = os.getcwd()
    # if eml_path not in sys.path:
    #    sys.path.insert(1, eml_path)
    main(sys.argv[1:])

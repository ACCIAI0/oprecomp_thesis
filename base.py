#!/usr/bin/python

import warnings
warnings.filterwarnings('ignore')

import sys
import os

from tensorflow.compat.v1 import logging

from utils.stopwatch import Stopwatch
import argsmanaging as am
import benchmarks
import training
import optimization
import utils.printing_utils as pu


def main(argv):
    stop_w = Stopwatch()

    # ArgumentsHolder containing all legal initialization params.
    args = am.handle_args(argv)
    print('[LOG] {}\n'.format(args))

    # Benchmarks information. Contains a relational graph among variables inside the benchmark program and the number
    # of them.
    stop_w.start()
    bm = benchmarks.get_benchmark(args.get__benchmark())
    bm.get_graph()
    _, t = stop_w.stop()
    print("[LOG] {} loaded in {:.3f}s ({} variables)"
          .format(pu.param(bm.get__name()), t, pu.accent_int(bm.get_vars_number())))

    # Build training set and test set for a new training session
    stop_w.start()
    session = training.create_training_session(args, bm, set_size=900)
    _, t = stop_w.stop()
    print("[LOG] Created first training session from dataset #{:d} in {:.3f}s ({} entries for training, "
          "{} for test)".format(args.get_dataset_index(), t, pu.accent_int(len(session.get_training_set())),
                                pu.accent_int(len(session.get_test_set()))))

    # Train a regressor
    trainer = training.RegressorTrainer.create_for(args.get_regressor(), session)
    stop_w.start()
    regressor = trainer.create_regressor(bm)
    _, t = stop_w.stop()
    print("[LOG] Regressor created in {:.3f}s".format(t))
    stop_w.start()
    trainer.train_regressor(regressor, verbose=False)
    r_stats = trainer.test_regressor(args, bm, regressor)
    _, t = stop_w.stop()
    print("[LOG] First training of the regressor completed in {:.3f}s (MAE {})"
          .format(t, pu.accent_f(r_stats['MAE'])))

    # Train a classifier
    stop_w.start()
    trainer = training.ClassifierTrainer.create_for(args.get_classifier(), session)
    classifier = trainer.create_classifier(bm)
    _, t = stop_w.stop()
    print("[LOG] Classifier created in {:.3f}s".format(t))
    stop_w.start()
    trainer.train_classifier(classifier)
    c_stats = trainer.test_classifier(args, bm, classifier)
    _, t = stop_w.stop()
    print("[LOG] First training of the classifier completed in {:.3f}s (accuracy {})"
          .format(t, pu.accent("{:.3f}%".format(c_stats['accuracy'] * 100))))

    # Create a MP model
    stop_w.start()
    optim_model = optimization.create_optimization_model(args, bm, regressor, classifier, limit_search_exp=4)
    _, t = stop_w.stop()
    print("[LOG] Created an optimization model in {:.3f}s\n".format(t))

    # Solve optimization problem
    config, its = optimization.try_model(args, bm, optim_model, regressor, classifier, stop_w,
                                         session)
    # TODO FINAL CHECK BEING... who knows

    print(pu.show("\nTotal execution time: {:.3f}s, refinement iterations: {:d}"
          .format(stop_w.get_duration(), its), alternate=True))


'''
Entry point. Imports EML and calls to main function if this is main module.
'''
if __name__ == '__main__':
    # Remove annoying warning prints from output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.set_verbosity(logging.ERROR)

    # Main call
    main(sys.argv[1:])

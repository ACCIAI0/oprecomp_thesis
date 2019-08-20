#!/usr/bin/python

import math

import pandas
import numpy

import benchmarks
import argsmanaging
from docplex.mp import model
from eml.backend import cplex_backend
from eml.tree import embed as dt_embed
from eml.tree.reader import sklearn_reader
from eml.net import process, embed as nn_embed
from eml.net.reader import keras_reader

__max_target_error = -numpy.log(1e-80)
__n_vars_bounds_tightening = 10
__bounds_opt_time_limit = 10
__classifier_threshold = .5


def __eml_regressor_nn(args: argsmanaging.ArgumentsHolder, benchmark: benchmarks.Benchmark, regressor):
    regressor_em = keras_reader.read_keras_sequential(regressor)
    regressor_em.reset_bounds()
    for neuron in regressor_em.layer(0).neurons():
        neuron.update_lb(args.minBitsNumber)
        neuron.update_lb(args.maxBitsNumber)

    process.ibr_bounds(regressor_em)

    if benchmark.get_vars_number() > __n_vars_bounds_tightening:
        bounds_backend = cplex_backend.CplexBackend()
        process.fwd_bound_tighthening(bounds_backend, regressor_em, timelimit=__bounds_opt_time_limit)

    return regressor_em


__eml_regressors = {
    argsmanaging.Regressor.NEURAL_NETWORK: __eml_regressor_nn
}


def __eml_classifier_dt(args: argsmanaging.ArgumentsHolder, benchmark: benchmarks.Benchmark, classifier):
    classifier_em = sklearn_reader.read_sklearn_tree(classifier)
    for attr in classifier_em.attributes():
        classifier_em.update_lb(attr, args.minBitsNumber)
        classifier_em.update_ub(attr, args.maxBitsNumber)

    return classifier_em


__eml_classifiers = {
    argsmanaging.Classifier.DECISION_TREE: __eml_classifier_dt
}


def create_optimization_model(args: argsmanaging.ArgumentsHolder, benchmark: benchmarks.Benchmark,
                              regressor, classifier, robustness: int = 0):
    backend = cplex_backend.CplexBackend()
    mod = model.Model()
    x_vars = [mod.integer_var(lb=args.minBitsNumber, ub=args.maxBitsNumber, name='x_{}'.format(i))
              for i in range(benchmark.get_vars_number())]
    y_var = mod.continuous_var(lb=-numpy.log(args.largeErrorThreshold), ub=__max_target_error, name='y')
    bit_sum_var = mod.integer_var(lb=args.minBitsNumber * benchmark.get_vars_number(),
                                  ub=args.maxBitsNumber * benchmark.get_vars_number(), name='bit_sum')
    class_var = mod.continuous_var(lb=0, ub=1, name='class')

    # Add relations from graph
    relations = benchmark.get_binary_relations()
    for vs, vg in relations['leq']:
        x_vs = mod.get_var_by_name('x_{}'.format(vs.get_index()))
        x_vg = mod.get_var_by_name('x_{}'.format(vg.get_index()))
        mod.add_constraint(x_vs <= x_vg)
    for vt, vv in relations['cast']:
        x_vt = mod.get_var_by_name('x_{}'.format(vt.get_index()))
        x_vv = [mod.get_var_by_name('x_{}'.format(v.get_index())) for v in vv]
        mod.add_constraint(mod.min(x_vv) == x_vt)

    reg_em = __eml_regressors[args.regressor](args, benchmark, regressor)
    nn_embed.encode(backend, reg_em, mod, x_vars, y_var, 'regressor')

    cls_em = __eml_classifiers[args.classifier](args, benchmark, classifier)
    dt_embed.encode_backward_implications(backend, cls_em, mod, x_vars, class_var, 'classifier')
    mod.add_constraint(class_var <= __classifier_threshold)
    mod.add_constraint(bit_sum_var == sum(x_vars))

    mod.minimize(mod.sum(x_vars))
    return mod

#!/usr/bin/python

import subprocess
from sys import float_info

import pandas
import numpy

import data_gen
from utils import io_utils
import benchmarks
from argsmanaging import args, Regressor, Classifier
import training
import utils.printing_utils as pu
from utils.stopwatch import stop_w

from docplex.mp import model
from eml.backend import cplex_backend
from eml.tree import embed as dt_embed
from eml.tree.reader import sklearn_reader
from eml.net import process, embed as nn_embed
from eml.net.reader import keras_reader

__max_target_error = numpy.ceil(-numpy.log(1e-80))
__n_vars_bounds_tightening = 10
__bounds_opt_time_limit = 10
__classifier_threshold = .5


class Iteration:
    def __init__(self, config, error, predicted_error, predicted_class, is_rollback,
                 previous_iteration=None):
        self.__config = config
        self.__error = error
        self.__pError = predicted_error
        self.__pClass = predicted_class
        self.__previous = previous_iteration
        self.__isRollback = is_rollback
        self.__args = args

    def get_config(self):
        return self.__config

    def get_error(self):
        return self.__error

    def get_error_class(self):
        return int(self.__error >= self.__args.get_large_error_threshold())

    def get_error_log(self):
        return -numpy.log(self.__error) if 0 != self.__error else float_info.min

    def get_predicted_error_log(self):
        return self.__pError

    def get_predicted_class(self):
        return self.__pClass

    def is_feasible(self):
        return self.get_error_log() >= self.__args.get_error_log()

    def get_delta_config(self):
        if self.__previous is None:
            return [0] * len(self.__config)
        return [i - j for i, j in zip(self.__previous.get_config(), self.__config)]

    def get_delta_error(self):
        if self.__previous is None:
            return 0
        return self.__previous.get_error() - self.__error

    def get_delta_error_log(self):
        if self.__previous is None:
            return 0
        return self.__previous.get_error_log() - self.get_error_log()

    def get_best_config(self):
        if self.__previous is not None:
            prev = self.__previous.get_best_config()
            if sum(self.__config) >= sum(prev) and self.__previous.is_feasible():
                return prev
        return self.__config

    def is_rollback(self):
        return self.__isRollback


def __eml_regressor_nn(bm: benchmarks.Benchmark, regressor):
    regressor_em = keras_reader.read_keras_sequential(regressor)
    regressor_em.reset_bounds()
    for neuron in regressor_em.layer(0).neurons():
        neuron.update_lb(args.get_min_bits_number())
        neuron.update_lb(args.get_max_bits_number())

    process.ibr_bounds(regressor_em)

    if bm.get_vars_number() > __n_vars_bounds_tightening:
        bounds_backend = cplex_backend.CplexBackend()
        process.fwd_bound_tighthening(bounds_backend, regressor_em, timelimit=__bounds_opt_time_limit)

    return regressor_em


__eml_regressors = {
    Regressor.NEURAL_NETWORK: __eml_regressor_nn
}


def __eml_classifier_dt(bm: benchmarks.Benchmark, classifier):
    classifier_em = sklearn_reader.read_sklearn_tree(classifier)
    for attr in classifier_em.attributes():
        classifier_em.update_lb(attr, args.get_min_bits_number())
        classifier_em.update_ub(attr, args.get_max_bits_number())

    return classifier_em


__eml_classifiers = {
    Classifier.DECISION_TREE: __eml_classifier_dt
}


def create_optimization_model(bm: benchmarks.Benchmark, regressor, classifier, robustness: int = 0, limit_search_exp=0):
    backend = cplex_backend.CplexBackend()
    mdl = model.Model()
    x_vars = [mdl.integer_var(lb=args.get_min_bits_number(), ub=args.get_max_bits_number(), name='x_{}'.format(i))
              for i in range(bm.get_vars_number())]

    # If is a limited search in n orders of magnitudes, change the upper_bound to be -log(10e(n + exp)) where exp is the
    # input parameter to calculate the error.
    upper_bound = __max_target_error
    if 0 != limit_search_exp:
        upper_bound = numpy.ceil(-numpy.log(numpy.float_power(10, -args.get_exponent() - limit_search_exp)))

    # Moved this to be a lower bound instead of a constraint
    target_error_log = numpy.ceil(args.get_error_log())
    max_config = pandas.DataFrame.from_dict({'var_{}'.format(i): [args.get_max_bits_number()]
                                             for i in range(bm.get_vars_number())})
    limit_predictable_error = regressor.predict(max_config)[0][0]
    if limit_predictable_error < target_error_log:
        robustness = -numpy.ceil(target_error_log - limit_predictable_error)

    y_var = mdl.continuous_var(lb=target_error_log + robustness, ub=upper_bound, name='y')

    bit_sum_var = mdl.integer_var(lb=args.get_min_bits_number() * bm.get_vars_number(),
                                  ub=args.get_max_bits_number() * bm.get_vars_number(), name='bit_sum')

    class_var = mdl.continuous_var(lb=0, ub=1, name='class')

    # Add relations from graph
    relations = bm.get_binary_relations()
    for vs, vg in relations['leq']:
        x_vs = mdl.get_var_by_name('x_{}'.format(vs.get_index()))
        x_vg = mdl.get_var_by_name('x_{}'.format(vg.get_index()))
        mdl.add_constraint(x_vs <= x_vg)
    for vt, vv in relations['cast']:
        x_vt = mdl.get_var_by_name('x_{}'.format(vt.get_index()))
        x_vv = [mdl.get_var_by_name('x_{}'.format(v.get_index())) for v in vv]
        mdl.add_constraint(mdl.min(x_vv) == x_vt)

    reg_em = __eml_regressors[args.get_regressor()](bm, regressor)
    nn_embed.encode(backend, reg_em, mdl, x_vars, y_var, 'regressor')

    cls_em = __eml_classifiers[args.get_classifier()](bm, classifier)
    dt_embed.encode_backward_implications(backend, cls_em, mdl, x_vars, class_var, 'classifier')
    mdl.add_constraint(class_var <= __classifier_threshold)
    mdl.add_constraint(bit_sum_var == sum(x_vars))

    mdl.minimize(mdl.sum(x_vars))

    mdl.set_time_limit(30)
    return mdl


def __refine_and_solve_mp(bm: benchmarks.Benchmark, mdl, n_iter, bad_config=None):

    # Remove an infeasible solution from the solutions pool
    if bad_config is not None:
        bin_vars_cut_vals = []
        for i in range(len(bad_config)):
            x_var = mdl.get_var_by_name('x_{}'.format(i))
            bin_vars_cut_vals.append(mdl.binary_var(name='bvcv_{}_{}'.format(n_iter, i)))
            mdl.add(mdl.if_then(x_var == bad_config[i], bin_vars_cut_vals[i] == 1))
        mdl.add_constraint(sum(bin_vars_cut_vals) <= 1)

    solution = mdl.solve()
    opt_config = None
    if solution is not None:
        opt_config = [int(solution['x_{}'.format(i)]) for i in range(bm.get_vars_number())]
    return opt_config, mdl


def __check_output(floating_result, target_result):
    very_large_error = 1000
    if len(floating_result) != len(target_result) or \
            all(v == 0 for v in floating_result) != all(v == 0 for v in target_result):
        return very_large_error

    sqnr = 0.00
    for i in range(len(floating_result)):
        # if floating_result[i] == 0, check_output returns 1: this is an
        # unwanted behaviour
        if floating_result[i] == 0.00:
            continue    # mmmhhh, TODO: fix this in a smarter way

        # if there is even just one inf in the result list, we assume that
        # for the given configuration the program did not run properly
        if str(floating_result[i]) == 'inf':
            return 'Nan'

        signal_sqr = target_result[i] ** 2
        error_sqr = (floating_result[i] - target_result[i]) ** 2
        temp = 0.00
        if error_sqr != 0.00:
            temp = signal_sqr / error_sqr
        if temp != 0:
            temp = 1.0 / temp
        if temp > sqnr:
            sqnr = temp

    return sqnr


def __run_check(program, target_result, dataset_index):
    params = [program, '42']
    if -1 != dataset_index:
        params.append(str(dataset_index))
    output = subprocess.Popen(params, stdout=subprocess.PIPE).communicate()[0]
    result = io_utils.parse_output(output.decode('utf-8'))
    return __check_output(result, target_result)


def __check_solution(bm: benchmarks.Benchmark, opt_config):
    io_utils.write_configs_file(bm.get_configs_file(), opt_config)
    if -1 == args.get_dataset_index():
        program = bm.get_home() + bm.get_map() + '.sh'
        target_file = bm.get_home() + 'target.txt'
    else:
        program = bm.get_home() + bm.get_map() + '_multiDataSet.sh'
        target_file = bm.get_home() + 'targets/target_{}.txt'.format(args.get_dataset_index())

    target_result = io_utils.read_target(target_file)
    error = __run_check(program, target_result, args.get_dataset_index())

    io_utils.write_configs_file(bm.get_configs_file(), [args.get_max_bits_number()] * bm.get_vars_number())
    return error


def __get_pred_class(config, regr, classifier):
    conf_df = pandas.DataFrame.from_dict({'var_{}'.format(i): [config[i]] for i in range(len(config))})
    prediction_with_conf = regr.predict(conf_df)[0]
    class_pred_with_conf = classifier.predict(conf_df)[0]
    return prediction_with_conf[0], class_pred_with_conf


def __iterate(bm: benchmarks.Benchmark, mdl, rollback_config, regressor, classifier, previous_it: Iteration = None,
              n_iter=0):
    bad_config = None
    if previous_it is not None and not previous_it.is_feasible():
        bad_config = previous_it.get_config()
    opt_config, mdl = __refine_and_solve_mp(bm, mdl, n_iter, bad_config=bad_config)
    is_rollback = opt_config is None
    if is_rollback:
        opt_config = rollback_config

    error = __check_solution(bm, opt_config)
    predicted_error, predicted_class = __get_pred_class(opt_config, regressor, classifier)
    return mdl, Iteration(opt_config, error, predicted_error, predicted_class, is_rollback, previous_it)


def __log_iteration(it: Iteration, n, t):
    print("[OPT] Solution # {:d} found in {:.3f}s:".format(n, t))

    target_label = "target error"
    error_label = "calculated error"
    predicted_label = "predicted error"
    log = "log({})"

    print("[OPT] {}  {}  |  {}  {}  {}"
          .format(pu.accent(error_label),
                  pu.accent(predicted_label),
                  pu.param(log.format(target_label)),
                  pu.accent(log.format(error_label)),
                  pu.accent(log.format(predicted_label))))

    pr = numpy.float_power(numpy.e, -it.get_predicted_error_log())

    l_target = (" " * (len(target_label) + 5 - pu.str_len_f(args.get_error_log()))) + pu.param_f(args.get_error_log())
    error = (" " * (len(error_label) - pu.str_len_e(it.get_error()))) + pu.accent_e(it.get_error())
    l_error = (" " * (len(error_label) + 5 - pu.str_len_f(it.get_error_log()))) + pu.accent_f(it.get_error_log())
    predicted = (" " * (len(predicted_label) - pu.str_len_e(pr))) + pu.accent_e(pr)
    l_predicted = (" " * (len(predicted_label) + 5 - pu.str_len_f(it.get_predicted_error_log()))) + \
                  pu.accent_f(it.get_predicted_error_log())
    print("[OPT] {}  {}  |  {}  {}  {}".format(error, predicted, l_target, l_error, l_predicted))

    if not it.is_rollback():
        ending = pu.accent("FEASIBLE", alternate=True) if it.is_feasible() else pu.err("NOT FEASIBLE")
        print("[OPT] Solution # {:d} {} is {}".format(n, pu.show(it.get_config()), ending))
    else:
        print("[OPT] {} {} {}".format(pu.warn("Solution # {:d}".format(n)), pu.show(it.get_config()),
                                      pu.warn("is a rollback. Next iteration")))


def try_model(bm: benchmarks.Benchmark, mdl, regressor, classifier, session: training.TrainingSession,
              max_iterations=100):
    stop_w.start()
    offset = int((args.get_max_bits_number() - args.get_min_bits_number()) / 2)
    mdl, it = __iterate(bm, mdl, [args.get_min_bits_number() + offset] * bm.get_vars_number(), regressor, classifier)
    _, t = stop_w.stop()
    __log_iteration(it, 0, t)
    print()

    iterations = 1
    while not (it.is_feasible() and not it.is_rollback()) and iterations <= max_iterations:
        stop_w.start()
        examples = data_gen.infer_examples(bm, it)
        _, t = stop_w.stop()
        print("[OPT] Inferred {} more examples in {:.3f}s".format(pu.accent_int(len(examples)), t))

        stop_w.start()
        session, r_stats, c_stats = data_gen.ml_refinement(bm, regressor, classifier, session, examples)
        _, t = stop_w.stop()
        print("[OPT] Retrained regressor (MAE {}) and classifier (accuracy {}) in {:.3f}s"
              .format(pu.accent_f(r_stats['MAE']), pu.accent("{:.3f}%".format(c_stats['accuracy'] * 100)), t))

        stop_w.start()
        mdl, it = __iterate(bm, mdl, [numpy.minimum(v + 1, args.get_max_bits_number()) for v in it.get_config()],
                            regressor, classifier, it, iterations)
        _, t = stop_w.stop()
        __log_iteration(it, iterations, t)
        print()
        iterations += 1

    return it.get_best_config(), iterations - 1

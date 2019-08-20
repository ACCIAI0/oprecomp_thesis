#!/usr/bin/python

import math
import decimal

import numpy
import pandas
from sklearn import preprocessing, metrics, tree
import keras
from keras import models, layers, callbacks

import argsmanaging as argsm
import benchmarks as bm


__dataset_home = "datasets/"

__NN_regressor_type = '_regrNN_2x1x'

__learning_rate = .001
__clamped_error_limit = .9


class TrainingSession:
    def __init__(self, training_set, test_set, regressor_target_label, classifier_target_label):
        self.__targetRegressor = training_set[regressor_target_label]
        self.__targetClassifier = training_set[classifier_target_label]

        del training_set[regressor_target_label]
        del training_set[classifier_target_label]

        self.__trainingSet = training_set

        self.__testRegressor = test_set[regressor_target_label]
        self.__testClassifier = test_set[classifier_target_label]

        del test_set[regressor_target_label]
        del test_set[classifier_target_label]

        self.__testSet = test_set

    def get_training_set(self):
        return self.__trainingSet

    def get_test_set(self):
        return self.__testSet

    def get_target_regressor(self):
        return self.__targetRegressor

    def get_target_classifier(self):
        return self.__targetClassifier

    def get_test_regressor(self):
        return self.__testRegressor

    def get_test_classifier(self):
        return self.__testClassifier


def __initialize_mean_std(benchmark: bm.Benchmark, label: str, log_label: str, clamp: bool = True):
    data_file = 'exp_results_{}.csv'.format(benchmark.get__name())

    df = pandas.read_csv(__dataset_home + data_file, sep=';')

    columns = [x for x in filter(lambda l: 'var_' in l or label == l, df.columns)]
    df = df[columns]

    if clamp:
        df.loc[df[label] > __clamped_error_limit, label] = __clamped_error_limit
    df[log_label] = [math.inf if 0 == x else -numpy.log(x) for x in df[label]]

    return df


def __select_subset(df: pandas.DataFrame, threshold: float, error_label: str, size: int):
    n_large_errors = len(df[df[error_label] >= threshold])
    ratio = n_large_errors / len(df)

    if 0 == ratio:
        return df.sample(size)
    acc = 0
    df_t = pandas.DataFrame()
    while acc < ratio:
        if size > len(df):
            size = len(df) - 1
        df_t = df.sample(size)
        acc = len(df_t[error_label] >= threshold) / len(df_t)
    return df_t


def create_training_session(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark,
                            initial_sampling_size: int = 3000, set_size: int = 500) -> TrainingSession:
    label = 'err_ds_{}'.format(args.datasetIndex)
    log_label = 'err_log_ds_{}'.format(args.datasetIndex)
    class_label = 'class_ds_{}'.format(args.datasetIndex)

    # Initialize a pandas DataFrame from file, clamping error values and calculating log errors
    df = __initialize_mean_std(benchmark, label, log_label)
    # Keep entries with all non-zero values
    df = df[(df != 0).all(1)]
    # Selects a subset with a balanced ratio between high and low error values
    df = __select_subset(df, args.largeErrorThreshold, label, initial_sampling_size)
    # Reset indexes to start from 0
    df = df.reset_index(drop=True)
    # Calculates the classifier class column
    df[class_label] = df.apply(lambda e: int(e[label] >= args.largeErrorThreshold), axis=1)
    # Delete err_ds_<index> column as it is useless from here on
    del df[label]

    return TrainingSession(df[:set_size], df[set_size:], log_label, class_label)


def __evaluate_predictions_regressor(predicted, actual):
    stats_res = {}
    pred_n_rows, pred_n_cols = predicted.shape
    pred = predicted[:, 0]
    if pred_n_cols > 1:
        mus = predicted[:, 0]
        sigmas = predicted[:, 1]
        predicted = mus
    else:
        predicted = pred

    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    underest_count = 0
    overest_count = 0
    errors = []

    for i in range(len(predicted)):
        abs_errors.append(abs(predicted[i] - actual[i]))
        errors.append(predicted[i] - actual[i])
        squared_errors.append((predicted[i] - actual[i]) *
                              (predicted[i] - actual[i]))
        if actual[i] != 0:
            p_abs_errors.append((abs(predicted[i] - actual[i])) *
                                100 / abs(actual[i]))
        sp_abs_errors.append((abs(predicted[i] - actual[i])) * 100 /
                             abs(predicted[i] + actual[i]))
        if predicted[i] - actual[i] > 0:
            overest_count += 1
        elif predicted[i] - actual[i] < 0:
            underest_count += 1

    stats_res["MAE"] = decimal.Decimal(numpy.mean(numpy.asarray(abs_errors)))
    stats_res["MSE"] = decimal.Decimal(numpy.mean(numpy.asarray(squared_errors)))
    stats_res["RMSE"] = decimal.Decimal(math.sqrt(stats_res["MSE"]))
    stats_res["MAPE"] = decimal.Decimal(numpy.mean(numpy.asarray(p_abs_errors)))
    stats_res["SMAPE"] = decimal.Decimal(numpy.mean(numpy.asarray(sp_abs_errors)))
    stats_res["ERRORS"] = errors
    stats_res["ABS_ERRORS"] = abs_errors
    stats_res["P_ABS_ERRORS"] = p_abs_errors
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors
    stats_res["SQUARED_ERRORS"] = squared_errors
    stats_res["R2"] = metrics.r2_score(actual, predicted)
    stats_res["MedAE"] = metrics.median_absolute_error(actual, predicted)
    stats_res["EV"] = metrics.explained_variance_score(actual, predicted)
    stats_res["abs_errs"] = abs_errors
    stats_res["p_abs_errs"] = p_abs_errors
    stats_res["sp_abs_errs"] = sp_abs_errors
    stats_res["squared_errs"] = squared_errors
    stats_res["accuracy"] = 100 - abs(stats_res["MAPE"])
    stats_res["underest_count"] = underest_count
    stats_res["overest_count"] = overest_count
    stats_res["underest_ratio"] = underest_count / len(predicted)
    stats_res["overest_ratio"] = overest_count / len(predicted)

    return stats_res


def __evaluate_predictions_classifier(predicted, actual):
    pred_classes = [0] * len(predicted)
    for i in range(len(predicted)):
        if .5 <= predicted[i]:
            pred_classes[i] = 1

    precision_s, recall_s, fscore_s, xyz = metrics.precision_recall_fscore_support(
        actual, pred_classes, average='binary', pos_label=0)
    precision_l, recall_l, fscore_l, xyz = metrics.precision_recall_fscore_support(
        actual, pred_classes, average='binary', pos_label=1)
    precision_w, recall_w, fscore_w, xyz = metrics.precision_recall_fscore_support(
        actual, pred_classes, average='weighted')

    stats_res = {
        'small_error_precision': precision_s,
        'small_error_recall': recall_s,
        'small_error_fscore': fscore_s,
        'large_error_precision': precision_l,
        'large_error_recall': recall_l,
        'large_error_fscore': fscore_l,
        'precision': precision_w,
        'recall': recall_w,
        'fscore': fscore_w,
        'accuracy': metrics.accuracy_score(actual, predicted)
    }

    return stats_res


def __train_regressor_nn(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, session: TrainingSession,
                         epochs: int = 100, batch_size: int = 32):
    # INITIALIZATION ===================================================================================================
    scaler = preprocessing.MinMaxScaler()
    train_data_tensor = scaler.fit_transform(session.get_training_set())
    test_data_tensor = scaler.fit_transform(session.get_test_set())
    train_target_tensor = scaler.fit_transform(session.get_target_regressor().values.reshape(-1, 1))
    test_target_tensor = scaler.fit_transform(session.get_test_regressor().values.reshape(-1, 1))
    n_samples, n_features = train_data_tensor.shape
    input_shape = (train_data_tensor.shape[1],)

    prediction_model = models.Sequential()

    if benchmark.get__name() == 'BlackSholes' or benchmark.get__name() == 'Jacobi':
        prediction_model.add(layers.Dense(int(n_features / 2), activation='relu', input_shape=input_shape))
        prediction_model.add(layers.Dense(int(n_features / 4), activation='relu'))
        prediction_model.add(layers.Dense(n_features, activation='relu'))
    elif __NN_regressor_type == '_regrNN_2x1x':
        prediction_model.add(layers.Dense(n_features * 2, activation='relu',
                                          activity_regularizer=keras.regularizers.l1(1e-5), input_shape=input_shape))
        prediction_model.add(layers.Dense(n_features, activation='relu'))

    prediction_model.add(layers.Dense(1, activation='linear'))
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-5)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2)

    adam = keras.optimizers.Adam(lr=__learning_rate)

    # TRAINING =========================================================================================================
    prediction_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    weights = numpy.full(len(train_data_tensor), 1)
    prediction_model.fit(train_data_tensor, train_target_tensor, sample_weight=weights, epochs=epochs,
                         batch_size=batch_size, shuffle=True, validation_split=0.1, verbose=True,
                         callbacks=[early_stopping, reduce_lr])

    # TESTING ==========================================================================================================
    test_loss = prediction_model.evaluate(test_data_tensor, test_target_tensor, verbose=0)
    predicted = prediction_model.predict(test_data_tensor)
    stats_res = __evaluate_predictions_regressor(predicted, test_target_tensor)
    stats_res['test_loss'] = test_loss

    max_conf = [args.maxBitsNumber for i in range(benchmark.get_vars_number())]
    max_conf_dict = {}
    for i in range(len(max_conf)):
        max_conf_dict['var_{}'.format(i)] = [max_conf[i]]
    max_conf_df = pandas.DataFrame.from_dict(max_conf_dict)
    stats_res['max_prediction_error'] = prediction_model.predict(max_conf_df)[0][0]

    return prediction_model, stats_res


def __train_classifier_dt(args: argsm.ArgumentsHolder, benchmark: bm.Benchmark, session: TrainingSession,
                          weight_large_errors: int = 10):
    classes = session.get_target_classifier().tolist()
    weights = [1] * len(classes)
    if weight_large_errors != 1:
        for i in range(len(classes)):
            if 1 == classes[i]:
                weights[i] = weight_large_errors
    classifier = tree.DecisionTreeClassifier(max_depth=benchmark.get_vars_number() + 5)
    classifier.fit(session.get_training_set(), session.get_target_classifier(), sample_weight=weights)
    predicted = classifier.predict(session.get_test_set())
    return classifier, __evaluate_predictions_classifier(predicted, session.get_test_classifier())


regressor_trainings = {
    argsm.Regressor.NEURAL_NETWORK: __train_regressor_nn
}

classifier_trainings = {
    argsm.Classifier.DECISION_TREE: __train_classifier_dt
}

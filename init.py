#!/usr/bin/python

from enum import Enum


class Regressor(Enum):
    NEURAL_NETWORK = 'NN'


class Classifier(Enum):
    DECISION_TREE = 'DT'


class Init:

    def __init__(self, benchmark=None, error=-1.0, regressor=Regressor.NEURAL_NETWORK,
                 classifier=Classifier.DECISION_TREE, dataset_index=0):
        self.benchmark = benchmark
        self.error = error
        self.regressor = regressor
        self.classifier = classifier
        self.datasetIndex = dataset_index

    def is_legal(self):
        return self.benchmark is not None and self.error is not -1

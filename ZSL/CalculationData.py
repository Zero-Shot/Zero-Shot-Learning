"""
Data used to calculate the new train and validation subsets
"""

from numpy.random import permutation
from math import floor


class CalculationData:
    def __init__(self, data, split):
        """
        :param data: ExperimentData object
        :param split: Split of training and validation. Float between 0 and 1
        """
        self._split = split
        self._train_classes = data.get_train_classes()
        self._train_indices = data.get_train_indices()
        self._train_labels = data.get_train_labels()
        self._random_permutation = permutation(len(self._train_classes))

    def get_classes(self):
        return self._train_classes

    def get_indices(self):
        return self._train_indices

    def get_labels(self):
        return self._train_labels

    def get_permutation(self):
        return self._random_permutation

    def get_training_classes(self):
        index_train_end = int(floor(len(self.get_classes()) * self._split))

        return self._random_permutation[:index_train_end]

    def get_validation_classes(self):
        index_train_end = int(floor(len(self.get_classes()) * self._split))
        class_len = len(self.get_classes())

        return self._random_permutation[index_train_end:class_len]

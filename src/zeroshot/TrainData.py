"""
Data used to train the parameters in the ZSL system
"""

import numpy as np

from zeroshot.CalculationData import CalculationData


class TrainData:
    def __init__(self, data, S):
        self._S_train = None
        self._S_validation = None

        self._training_indices = []
        self._training_labels = []
        self._training_classes = []

        self._validation_indices = []
        self._validation_labels = []
        self._validation_classes = []

        self.create_data(data, S)

    def get_S_train(self):
        return self._S_train

    def get_S_val(self):
        return self._S_validation

    def get_training_indices(self):
        return self._training_indices

    def get_training_labels(self):
        return self._training_labels

    def get_training_classes(self):
        return self._training_classes

    def get_validation_indices(self):
        return self._validation_indices

    def get_validation_labels(self):
        return self._validation_labels

    def get_validation_classes(self):
        return self._validation_classes

    @staticmethod
    def __create_indices(indices, labels, class_label):
        new_indices = []

        for train_index in range(len(indices)):
            if labels[train_index] == class_label:
                new_indices.append(indices[train_index])

        return new_indices

    def create_from_classes(self,
                            class_selection,
                            train_indices,
                            train_labels,
                            permutation,
                            S_train,
                            index_addition=0):
        """
        Calculate the new S matrix, indices and labels for the given selection of classes
        :param class_selection: part of the original train classes
        :param train_indices: original train indices
        :param train_labels: original train labels
        :param permutation: random permutation generated
        :param S_train: original S_train matrix
        :param index_addition: added shift for validation set
        :return: S matrix, indices, labels
        """
        attributes = []
        created_indices = []
        created_labels = []

        for index in range(len(class_selection)):
            class_label = permutation[index_addition + index]
            new_indices = self.__create_indices(train_indices, train_labels, class_label)

            attributes.append(S_train[class_label, :])
            created_indices.extend(new_indices)
            created_labels.extend([index] * len(new_indices))

        S = np.array(attributes)

        return S, created_indices, created_labels

    def create_data(self, data, S):
        """
        Create the new data for training
        :param data: ExperimentData object
        :param S: original S matrix
        """
        calculation_data = CalculationData(data, 0.8)

        # Create new train and validation split

        self._training_classes = calculation_data.get_training_classes()
        self._validation_classes = calculation_data.get_validation_classes()

        # Create the new S matrices, indices and labels

        self._S_train, self._training_indices, self._training_labels = self. \
            create_from_classes(self._training_classes,
                                calculation_data.get_indices(),
                                calculation_data.get_labels(),
                                calculation_data.get_permutation(),
                                S)

        self._S_validation, self._validation_indices, self._validation_labels = self. \
            create_from_classes(self._validation_classes,
                                calculation_data.get_indices(),
                                calculation_data.get_labels(),
                                calculation_data.get_permutation(),
                                S,
                                len(self._training_classes))

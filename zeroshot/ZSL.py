"""
Zero-Shot learning system
"""

import scipy as sp
import numpy as np
import os

from zeroshot.ExperimentData import ExperimentData
from zeroshot.TrainData import TrainData
from zeroshot.Parameters import Parameters
from zeroshot.ExperimentResults import ExperimentResults
from sklearn.metrics.pairwise import pairwise_distances


class ZSL:
    _X_TRAIN_FILENAME = "Xtrain"
    _X_TEST_FILENAME = "Xtest"
    _S_FILENAME = "Smatrix"
    _LOG_OUTPUT = True

    def __init__(self, dataset_path, log=True):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Path %s not found" % dataset_path)

        self._dataset_path = dataset_path
        self._kernel_sigmas = [0]  # 0 indicates the Linear Kernel
        self._gammas = [1]
        self._lambdas = [1]
        self._parameters = Parameters()
        self._LOG_OUTPUT = log

        self._data = ExperimentData()
        self._X = self.__read_X()
        self._S, self._class_names = self.__read_S_file()
        self._K = None

    def set_parameters(self, kernel_sigmas, gammas, lambdas):
        """
        Set parameters of the system
        :param kernel_sigmas: List of sigmas for the kernel calculation, 0 is for Linear Kernel
        :param gammas: List of gamma values for the V matrix
        :param lambdas: List of lambda values for the V matrix
        """
        self._kernel_sigmas = kernel_sigmas
        self._gammas = gammas
        self._lambdas = lambdas

    def __log(self, text):
        if self._LOG_OUTPUT:
            print(text)

    @staticmethod
    def __read_X_from_file(X, filename, indices, labels, classes, idx=0):
        """
        Add to X matrix, indices, labels and classes from given file
        :param X: X list to add to
        :param filename: File to read from
        :param indices: List of indices to add to
        :param labels: List of labels to add to
        :param classes: List of classes to add to
        :param idx: start value of indices
        :return:
        """
        for line in open(filename):
            line_list = line.split()
            class_id = int(line_list[0])
            feature_vector = list(map(float, line_list[1:]))

            indices.append(idx)
            labels.append(class_id)

            if class_id not in classes:
                classes.append(class_id)

            X.append(feature_vector)
            idx += 1

    def __create_gaussian_kernel(self, sigma):
        """
        Set K to Gaussian K kernel, from X using sigma
        :param sigma: parameter for the Gaussian calculation
        """
        pairwise_dists = pairwise_distances(self._X, self._X[self._data.get_train_indices()], metric='euclidean', n_jobs=-1)
        self._K = sp.exp(-pairwise_dists ** 2 / sigma ** 2)

    def __create_linear_kernel(self):
        """
        Set K to Linear Kernel, from X
        """
        self._K = self._X @ self._X[self._data.get_train_indices()].T

    def __read_X(self):
        """
        Read in the X matrix and ExperimentData from two X files: Xtrain and Xtest
        :return: numpy array of the X matrix
        """
        X = []
        train_indices = []
        train_labels = []
        train_classes = []
        test_indices = []
        test_labels = []
        test_classes = []

        # Read the train and test X matrix files
        self.__read_X_from_file(X,
                                os.path.join(self._dataset_path, self._X_TRAIN_FILENAME),
                                train_indices,
                                train_labels,
                                train_classes)
        self.__read_X_from_file(X,
                                os.path.join(self._dataset_path, self._X_TEST_FILENAME),
                                test_indices,
                                test_labels,
                                test_classes,
                                len(train_indices))

        # Set obtained indices, labels and classes for train and test
        self._data.set_data(train_indices,
                            train_labels,
                            train_classes,
                            test_indices,
                            test_labels,
                            test_classes)

        return np.array(X)

    def __read_S_file(self):
        """
        Read the S matrix file and return the S matrix and the class names
        :return: numpy array of S matrix, list of class names
        """
        S = []
        class_names = []

        for line in open(os.path.join(self._dataset_path, self._S_FILENAME)):
            line_list = line.split(',')
            class_names.append(line_list[0])
            S.append(list(map(float, line_list[1:])))

        return np.array(S), class_names

    @staticmethod
    def __calculate_Y(instances_indices, classes_indices, instances_labels):
        """
        Create and return the Y matrix, used to mark the corresponding class label to the instances
        :param instances_indices:
        :param classes_indices:
        :param instances_labels:
        :return: numpy array of Y matrix
        """
        Y = np.zeros((len(instances_indices),
                      len(classes_indices)))

        for idx in range(len(instances_indices)):
            Y[idx, instances_labels[idx]] = 1

        return Y

    def __calculate_V(self, K, Y, S):
        """
        Calculate the translation matrix V
        :param K: Kernel matrix K
        :param Y: Matrix of the class labeled per instance
        :param S: Attribute matrix S
        :return:
        """
        KK = K.T @ K
        KYS = K @ Y @ S
        KYS_invSS = KYS @ sp.linalg.inv(S.T @ S + self._parameters.get_lambda() * np.identity(np.shape(S)[1]))

        return sp.linalg.inv(KK + self._parameters.get_gamma() * np.identity(np.shape(K)[1])) @ KYS_invSS

    @staticmethod
    def __predict_classes(K, V, S):
        """
        Predict the classes of the instances
        :param K: Test Kernel
        :param V: The translation matrix V
        :param S: Test attribute matrix S
        :return: List of class predictions
        """
        prediction_matrix = K @ V @ S.T

        class_prediction = []

        for instance_columns in prediction_matrix:
            # Find the highest achieving prediction per instance
            class_max = max(instance_columns)
            class_number = instance_columns.tolist().index(class_max)

            class_prediction.append(class_number)

        return class_prediction

    @staticmethod
    def __evaluate(class_predictions, labels):
        correct = 0

        for prediction, instance_label in zip(class_predictions, labels):
            if prediction == instance_label:
                correct += 1

        return correct / len(class_predictions)

    def check_record(self, performance, record):
        if performance > record:
            self.__log("New Record: " + str(performance))
            return True
        return False

    def train(self, S_train):
        """
        Train the system on the parameters
        :param S_train: S matrix for the train set
        """
        self.__log("Optimising parameters (gaussian_sigma, gamma, lambda)...")

        # Create new training data split, containing train and validation
        train_data = TrainData(self._data, S_train)
        S_val = train_data.get_S_val()
        record = 0

        for sigma in self._kernel_sigmas:
            # Create the new Kernel K
            if sigma == 0:
                self.__create_linear_kernel()
            else:
                self.__create_gaussian_kernel(sigma)

            K_train = self._K[train_data.get_training_indices()][:, train_data.get_training_indices()]
            KK = K_train.T @ K_train

            Y = self.__calculate_Y(train_data.get_training_indices(),
                                   train_data.get_training_classes(),
                                   train_data.get_training_labels())

            S = train_data.get_S_train()
            SS = S.T @ S
            KYS = K_train @ Y @ S

            for gamma in self._gammas:
                invKK_KYS = sp.linalg.inv(KK + gamma * np.identity(np.shape(K_train)[1])) @ KYS

                for lambda_val in self._lambdas:
                    V = invKK_KYS @ sp.linalg.inv(SS + lambda_val * np.identity(np.shape(S)[1]))

                    K_val = self._K[train_data.get_validation_indices()][:, train_data.get_training_indices()]

                    # Calculate class predictions
                    class_prediction = self.__predict_classes(K_val, V, S_val)

                    # Store current performance
                    current_performance = self.__evaluate(class_prediction, train_data.get_validation_labels())

                    # If performance is better than the record, store the current settings
                    if self.check_record(current_performance, record):
                        record = current_performance
                        self._parameters.set_lambda(lambda_val)
                        self._parameters.set_gamma(gamma)
                        self._parameters.set_sigma(sigma)

    def run(self):

        S_train = self._S[self._data.get_train_classes(), :]
        S_test = self._S[self._data.get_test_classes(), :]

        # Train the system for optimal parameters
        self.train(S_train)

        self.__log("Chosen parameters: " + str(self._parameters))

        # Calculate Kernel K
        if self._parameters.get_sigma() == 0:
            self.__create_linear_kernel()
        else:
            self.__create_gaussian_kernel(self._parameters.get_sigma())

        Y = self.__calculate_Y(self._data.get_train_indices(),
                               self._data.get_train_classes(),
                               self._data.get_train_labels())

        K_train = self._K[self._data.get_train_indices()][:, self._data.get_train_indices()]
        K_test = self._K[self._data.get_test_indices()][:, self._data.get_train_indices()]

        V = self.__calculate_V(K_train, Y, S_train)

        # Calculate class prediction of test set
        class_prediction = self.__predict_classes(K_test, V, S_test)

        # Calculate accuracy
        accuracy = self.__evaluate(class_prediction, self._data.get_test_labels())

        # Log accuracy
        self.__log("Final Score: %f" % accuracy)

        return ExperimentResults(accuracy,
                                 class_prediction,
                                 self._data.get_test_labels(),
                                 self._parameters,
                                 self._data.get_test_classes(),
                                 self._class_names)

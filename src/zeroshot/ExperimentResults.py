"""
Results of the ZSL experiment
"""

import pandas as pd
import numpy as np


class ExperimentResults:
    def __init__(self, accuracy, predictions, labels, parameters, test_classes, class_names):
        self._accuracy = accuracy
        self._predictions = predictions
        self._labels = labels
        self._parameters = parameters
        self._test_classes = test_classes
        self._class_names = class_names

    def get_accuracy(self):
        return self._accuracy

    def get_predictions(self):
        return self._predictions

    def get_labels(self):
        return self._labels

    def __get_class_name(self, label):
        return self._class_names[self._test_classes[label]]

    def get_confusion_matrix(self):
        confusion_dict = {}

        # Count number of instances and correct classifications per label
        for index, label in enumerate(self._labels):
            if label not in list(confusion_dict.keys()):
                confusion_dict[label] = {"class name": self.__get_class_name(label),
                                         "correct": 0,
                                         "total": 0,
                                         "percentage correct": 0,
                                         "percentage false": 0}

            confusion_dict[label]["total"] += 1

            # If classified correctly, increase count
            if label == self._predictions[index]:
                confusion_dict[label]["correct"] += 1

        # Calculate percentage correct and false
        for k, v in confusion_dict.items():
            confusion_dict[k]["percentage correct"] = v["correct"] * 100 / v["total"]
            confusion_dict[k]["percentage false"] = 100 - confusion_dict[k]["percentage correct"]

        # Put in pandas data frame and transpose
        return pd.DataFrame(confusion_dict).T

    def print_confusion_matrix(self):
        df = self.get_confusion_matrix()

        print(df.loc[:, ["class name", "percentage correct", "percentage false"]])

    def get_prediction_results(self):
        results = pd.DataFrame(columns=["label", "prediction"])

        for index in range(len(self._predictions)):
            results.loc[index] = [self.__get_class_name(self._labels[index]),
                                  self.__get_class_name(self._predictions[index])]

        return results

    def save_accuracy_to_file(self, file_path):
        """
        Store accuracy and parameters to output txt
        :param file_path:
        :return:
        """
        with open(file_path, "a") as file:
            file.write("Parameters: " + str(self._parameters) + ". Final score: " + str(self._accuracy) + '\n')

    def get_prediction_matrix(self):
        df = pd.DataFrame(np.zeros((len(self._test_classes), len(self._test_classes))))

        for index, label in enumerate(self._labels):
            row = label
            col = self._predictions[index]
            df[col][row] += 1

        df.columns = np.array(self._class_names)[self._test_classes]
        df.index = np.array(self._class_names)[self._test_classes]

        return df

    def save_prediction_matrix_to_file(self, file_path):
        self.get_prediction_matrix().to_csv(file_path)


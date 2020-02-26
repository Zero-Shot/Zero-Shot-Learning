"""
Dictionary of data which is used to run the ZSL experiment
"""


class ExperimentData:
    TRAIN_INDICES = "train_indices"
    TRAIN_CLASSES = "train_classes"
    TRAIN_LABELS = "train_labels"

    TEST_INDICES = "test_indices"
    TEST_CLASSES = "test_classes"
    TEST_LABELS = "test_labels"

    def __init__(self):
        self.__dictionary = dict()

        self.__dictionary[self.TRAIN_INDICES] = []
        self.__dictionary[self.TRAIN_LABELS] = []
        self.__dictionary[self.TRAIN_CLASSES] = []

        self.__dictionary[self.TEST_INDICES] = []
        self.__dictionary[self.TEST_LABELS] = []
        self.__dictionary[self.TEST_CLASSES] = []

    @staticmethod
    def create_indices(indices, labels, class_label):
        new_indices = []

        for index in range(len(indices)):
            if labels[index] == class_label:
                new_indices.append(indices[index])

        return new_indices

    def create_from_classes(self, classes, indices, labels):
        created_indices = []
        created_labels = []

        for index in range(len(classes)):
            class_label = classes[index]
            new_indices = self.create_indices(indices, labels, class_label)
            created_indices.extend(new_indices)
            created_labels.extend([index] * len(new_indices))

        return created_indices, created_labels

    def set_data(self, train_indices, train_labels, train_classes, test_indices, test_labels, test_classes):
        self[self.TRAIN_CLASSES] = train_classes
        self[self.TEST_CLASSES] = test_classes

        self[self.TRAIN_INDICES], self[self.TRAIN_LABELS] = self.create_from_classes(
            train_classes,
            train_indices,
            train_labels
        )

        self[self.TEST_INDICES], self[self.TEST_LABELS] = self.create_from_classes(
            test_classes,
            test_indices,
            test_labels
        )

    def get_train_indices(self):
        return self[ExperimentData.TRAIN_INDICES]

    def get_train_labels(self):
        return self[ExperimentData.TRAIN_LABELS]

    def get_train_classes(self):
        return self[ExperimentData.TRAIN_CLASSES]

    def get_test_indices(self):
        return self[ExperimentData.TEST_INDICES]

    def get_test_labels(self):
        return self[ExperimentData.TEST_LABELS]

    def get_test_classes(self):
        return self[ExperimentData.TEST_CLASSES]

    def __getitem__(self, item):
        return self.__dictionary[item]

    def __setitem__(self, key, value):
        self.__dictionary[key] = value

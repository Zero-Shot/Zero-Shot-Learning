"""
Parameter class containing all the parameters for the ZSL system
"""


class Parameters:
    PARAMETER_KEY_GAMMA = "gamma"
    PARAMETER_KEY_LAMBDA = "lambda"
    PARAMETER_KEY_SIGMA = "sigma"

    def __init__(self):
        self.__parameters = dict()

    def get_gamma(self):
        return self[self.PARAMETER_KEY_GAMMA]

    def get_lambda(self):
        return self[self.PARAMETER_KEY_LAMBDA]

    def get_sigma(self):
        return self[self.PARAMETER_KEY_SIGMA]

    def set_gamma(self, gamma):
        self[self.PARAMETER_KEY_GAMMA] = gamma

    def set_lambda(self, lambda_val):
        self[self.PARAMETER_KEY_LAMBDA] = lambda_val

    def set_sigma(self, sigma):
        self[self.PARAMETER_KEY_SIGMA] = sigma

    def __getitem__(self, item):
        return self.__parameters[item]

    def __setitem__(self, key, value):
        self.__parameters[key] = value

    def __str__(self):
        return str(self.__parameters)

"""
Example of a run script for the system
"""

import sys
import os
from ZSL.ZSL import ZSL


def run_arg(path_name, number_of_runs):
    zsl = ZSL(path_name)

    for idx in range(number_of_runs):
        zsl.set_parameters([0],
                           [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                           [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
        result = zsl.run()

        conf_matrix = result.get_confusion_matrix()
        print(conf_matrix)
        pred_results = result.get_prediction_results()
        print(pred_results)
        result.save_prediction_matrix_to_file(os.path.join(path_name, "matrix.csv"))

        result.save_accuracy_to_file(os.path.join(path_name, "output.txt"))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        run_arg(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) > 1:
        run_arg(sys.argv[1], 1)
    else:
        exit("Path to data files missing")

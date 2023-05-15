########################################################################
#
#	This script is part of the code for the characteristic mapping method in 2D with particle flow
#	written in C++ (C) using Nvidia CUDA on Linux.
#
#   The code is managed under GNU General Public License v3.0. Everyone is permitted to copy
#   and distribute verbatim copies of this license document, but changing it is not allowed.
#
#   Documentation and further information can be taken from the GitHub page, located at:
#   https://github.com/CharacteristicMappingMethod/cmm-turbulence
#
########################################################################

import os
import test_functions as test  # load testing functions
import importlib  # load individual test files

########################################################################
#   Change settings here
########################################################################

# list of tests to be executed
test_names = [
    "euler2D_11b_4nodes_reference_remapping"
]

########################################################################

def run_test_list():
    for i_test in test_names:
        # load test configs
        i_load = importlib.import_module(f"{i_test}.run_test")  # load with f-string, as we need dot for subfolder
        i_params = test.TestParams(i_load.test_param_dict)
        test.parse_command_line(i_params)
        compare_pass = test.run_test(i_params)


if __name__ == "__main__":
    run_test_list()
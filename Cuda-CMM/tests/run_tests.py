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
    ["euler2D_init_fields", "test_param_dict"],  # check for initialization of euler2D w particles
    ["euler2D_one_step", "test_param_dict"],  # check for computing one time-step of euler2D w particles
    ["euler2D_11b_4nodes_reference_remapping", "test_param_dict"],  # check for reference simulation including one remapping and w particles
    ["vlasov1D_15_vp_landau_damping", "test_param_dict"]  # vlasov test
]

########################################################################

def run_test_list():
    for i_test in test_names:
        # load test configs
        i_load = importlib.import_module(f"{i_test[0]}.run_test")  # load with f-string, as we need dot for subfolder
        i_params = test.TestParams(getattr(i_load, i_test[1]))
        test.parse_command_line(i_params)
        compare_pass = test.run_test(i_params)


if __name__ == "__main__":
    run_test_list()
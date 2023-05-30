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

import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import test_functions as test

########################################################################
#   Change test specific settings here
########################################################################

test_param_dict = {
    "name": "euler2D_11b",  # Specific name of test
    "wkdir": "tests/euler2D_11b_4nodes_reference_remapping",  # relative path from root to data folder of this test
    "param_loc": ["examples/euler2D/params-11b-4nodes-reference-remapping.txt"],  # params file relative from root path
    "results_loc": ["11b-4nodes-reference-remapping_4_nodes_C256_F512_t32_T2"],  # name of newly computed results
    "reference_loc": "reference-results_4_nodes_C256_F512_t32_T2"  # name of reference results
}

########################################################################

# executing this file directly runs only this test
if __name__ == "__main__":

    params = test.TestParams(test_param_dict)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)
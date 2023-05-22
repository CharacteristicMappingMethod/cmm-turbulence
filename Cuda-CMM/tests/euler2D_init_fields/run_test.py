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

import os, shutil, sys
from subprocess import PIPE, Popen  # run system commands
import logging
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import test_functions as test

########################################################################
#   Change test specific settings here
########################################################################

test_param_dict_none = {
    "name": "euler2D_init_fields_none",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_none.txt",  # params file relative from root path
    "results_loc": "init_fields_none_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_none_4_nodes_C256_F512_t1_T0"  # name of reference results
}
test_param_dict_comp = {
    "name": "euler2D_init_fields_comp",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_comp.txt",  # params file relative from root path
    "results_loc": "init_fields_comp_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_comp_4_nodes_C256_F512_t1_T0"  # name of reference results
}
test_param_dict_conv = {
    "name": "euler2D_init_fields_conv",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_conv.txt",  # params file relative from root path
    "results_loc": "init_fields_conv_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_conv_4_nodes_C256_F512_t1_T0"  # name of reference results
}
test_param_dict_sample = {
    "name": "euler2D_init_fields_sample",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_sample.txt",  # params file relative from root path
    "results_loc": "init_fields_sample_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_sample_4_nodes_C256_F512_t1_T0"  # name of reference results
}
test_param_dict_zoom = {
    "name": "euler2D_init_fields_zoom",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_zoom.txt",  # params file relative from root path
    "results_loc": "init_fields_zoom_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_zoom_4_nodes_C256_F512_t1_T0"  # name of reference results
}
test_param_dict_particles = {
    "name": "euler2D_init_fields_particles",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_particles.txt",  # params file relative from root path
    "results_loc": "init_fields_particles_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_particles_4_nodes_C256_F512_t1_T0"  # name of reference results
}
test_param_dict_forward = {
    "name": "euler2D_init_fields_forward",  # Specific name of test
    "wkdir": "tests/euler2D_init_fields",  # relative path from root to data folder of this test
    "param_loc": "tests/euler2D_init_fields/params_forward.txt",  # params file relative from root path
    "results_loc": "init_fields_forward_4_nodes_C256_F512_t1_T0",  # name of newly computed results
    "reference_loc": "reference-results_forward_4_nodes_C256_F512_t1_T0"  # name of reference results
}

########################################################################

# executing this file directly runs all tests from this script
if __name__ == "__main__":

    params = test.TestParams(test_param_dict_none)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)

    params = test.TestParams(test_param_dict_comp)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)

    params = test.TestParams(test_param_dict_conv)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)

    params = test.TestParams(test_param_dict_sample)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)

    params = test.TestParams(test_param_dict_zoom)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)

    params = test.TestParams(test_param_dict_particles)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)

    params = test.TestParams(test_param_dict_forward)
    test.parse_command_line(params)
    params.root_path = "../../"  # we have to adapt the root path
    compare_pass = test.run_test(params)
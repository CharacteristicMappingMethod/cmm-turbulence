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


########################################################################
#   Change settings here
########################################################################

root_path = "../../"  # relative root path from execution directory to Cuda-CMM
exec_loc = "SimulationCuda2d.out"  # executable relative from root path
param_loc = "examples/euler2D/params-11b-4nodes-reference-remapping.txt"  # params file relative from root path
wkdir = "tests/euler2D-11b-4nodes-reference-remapping"
debug_level = 0 # 0 - error and final, 1 - individual arrays and more

########################################################################


import os, shutil
from subprocess import PIPE, Popen  # run system commands
import logging
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)


# run simulation
if debug_level > 0: logging.info("Starting simulation")
process = Popen(
    [os.path.join(root_path, exec_loc), 'param_file=' + os.path.join(root_path, param_loc),
        'workspace=' + os.path.join(root_path, wkdir) + "/"
    ], stdout=PIPE)
output, _error = process.communicate()
if debug_level > 0: logging.info("Concluded simulation")


def read_mesure(loc):
    res_return = {}
    for i_res in os.listdir(os.path.join(root_path, wkdir, loc)):
        if i_res.startswith("Hash"):
            res_read = np.fromfile(os.path.join(root_path, wkdir, loc, i_res), dtype=np.uint8)
            res_hex = [hex(i_res_in)[2:].upper() for i_res_in in res_read]
            res_join = ''.join(res_hex)
            res_return[i_res] = [res_join[i:i+32] for i in range(0, len(res_join), 32)]
        else: res_return[i_res] = np.fromfile(os.path.join(root_path, wkdir, loc, i_res), dtype=float)
    return res_return

# load in simulation data
if debug_level > 0: logging.info("Loading simulation results")
results_loc = "data/11b-4nodes-reference-remapping_4_nodes_C256_F512_t32_T2/Monitoring_data/Mesure"
res_in = read_mesure(results_loc)

# load in reference data
if debug_level > 0: logging.info("Loading reference results")
reference_loc = "data/reference-results_4_nodes_C256_F512_t32_T2/Monitoring_data/Mesure"
res_ref = read_mesure(reference_loc)

# compare data
compare_pass = 0  # counter for failed tests
for i_res in os.listdir(os.path.join(root_path, wkdir, results_loc)):
    if not i_res in res_ref.keys():
        logging.error(f"Data {i_res} not available for reference results")
    else:
        if i_res.startswith("Hash"):
            diff = [res_in[i_res][i_len] == res_ref[i_res][i_len] for i_len in range(len(res_ref[i_res]))]
            if sum(diff) == len(res_ref[i_res]):
                if debug_level > 0: logging.info(f'{i_res}: Passed')
            else:
                logging.error(f'{i_res}: Failed with matching {diff} for times {res_in["Time_s_256.data"]}')
                compare_pass += 1
        elif i_res.startswith("Time"):
            diff = res_in[i_res] - res_ref[i_res]
            sum_diff = np.sum(diff)
            if sum_diff != 0:
                logging.error(f'{i_res}: Timings are not equal. This simulation: {res_in["Time_s.data"]}, reference: {res_ref["Time_s.data"]}')
                compare_pass += 1
        else:
            diff = res_in[i_res] - res_ref[i_res]
            sum_diff = np.sum(diff)
            if sum_diff == 0:
                if debug_level > 0: logging.info(f'{i_res}: Passed')
            else:
                logging.error(f'{i_res}: Failed with difference {diff} for times {res_in["Time_s.data"]}')
                compare_pass += 1

# remove data from simulation
# logging.info("Cleaning simulation data")
# shutil.rmtree(os.path.join(root_path, wkdir, "data/11b-4nodes-reference-remapping_4_nodes_C256_F512_t32_T2"))

# final verdict
logging.info(f'Completed test with {compare_pass} mismatches')
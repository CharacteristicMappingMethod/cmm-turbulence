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

import os, shutil
import logging
import numpy as np
import argparse
import re  # check for number in name to differ sampling from computaional mesure data
from subprocess import PIPE, Popen  # run system commands


logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%d-%m-%y %H:%M:%S',
    level=logging.INFO
)


# class for storing all the test parameters
class TestParams:
    # locations
    root_path = "../"  # relative root path from execution directory to Cuda-CMM
    exec_loc = "SimulationCuda2d.out"  # executable relative from root path

    # specific settings
    debug_level = 0  # 0 - error and final, 1 - individual arrays and more
    same_architecture = False  # True - compare hashes as well
    save_reference = False  # True - overwrite reference results if test passed
    save_force = False  # True - overwrite reference results
    eps_pass = 1e-13  # everything smaller is considered equal

    # init function sets test specific parameters
    def __init__(self, str_dict):
        self.name = str_dict['name']  # Specific name of test
        self.wkdir = str_dict['wkdir']  # relative path from root to data folder of this test
        self.param_loc = str_dict['param_loc']  # params file relative from root path
        self.results_loc = str_dict['results_loc']  # name of newly computed results
        self.reference_loc = str_dict['reference_loc']  # name of reference results
    
    def logInfo(self, text: str):
        if self.debug_level > 0: logging.info(f"{self.name} - {text}")
    def logError(self, text: str):
        if self.debug_level >= 0: logging.error(f"{self.name} - {text}")


# command line options
def parse_command_line(params: TestParams):
    parser = argparse.ArgumentParser(description='Script for testing cmm')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output of tests')
    parser.add_argument('-a', '--architecture', action='store_true', help='Include hashing in tests, only works on same machine as reference values')
    parser.add_argument('-r', '--reference', action='store_true', help='Save Simulation results as new reference if they passed the test')
    parser.add_argument('-f', '--force', action='store_true', help='Save Simulation results as new reference even if they failed the test')
    args = parser.parse_args()

    if args.verbose: params.debug_level = 1
    if args.architecture: params.same_architecture = True
    if args.reference: params.save_reference = True
    if args.force: params.save_force = True


# run simulation
# root_path     - rel path from executed file to main path of cmm
# wkdir         - path of executed test, where the data should be stored
# param_loc     - location of used parameter file
def run_sim(params: TestParams):
    # run simulation
    params.logInfo("Starting simulation")
    process = Popen(
        [os.path.join(params.root_path, params.exec_loc), 'param_file=' + os.path.join(params.root_path, params.param_loc),
            'workspace=' + os.path.join(params.root_path, params.wkdir) + "/"
        ], stdout=PIPE)
    output, _error = process.communicate()
    params.logInfo("Concluded simulation")


# function to read in monitoring and mesure values
# root_path     - rel path from executed file to main path of cmm
# wkdir         - path of executed test
# loc           - name of data to be loaded
def read_monitoring(params: TestParams, loc):
    params.logInfo(f"Loading monitoring results: {loc}")   
    res_return, loc_data = [{}, {}], [os.path.join("data", loc, "Monitoring_data"), os.path.join("data", loc, "Monitoring_data", "Mesure")]
    for i_res, i_loc in zip(res_return, loc_data):
        for i_load in os.listdir(os.path.join(params.root_path, params.wkdir, i_loc)):
            if i_load.endswith(".data"):
                if i_load.startswith("Hash"):
                    res_read = np.fromfile(os.path.join(params.root_path, params.wkdir, i_loc, i_load), dtype=np.uint8)
                    res_join = ''.join([hex(i_res_in)[2:].upper() for i_res_in in res_read])
                    i_res[i_load] = [res_join[i:i+32] for i in range(0, len(res_join), 32)]
                else: i_res[i_load] = np.fromfile(os.path.join(params.root_path, params.wkdir, i_loc, i_load), dtype=float)
    return res_return


# function to compare mesure values:
# Time values               - Mismatches can in theory only occur from different save settings
# Global quantitities       - Mismatches are compared up to precision of eps_pass
# Hash quantitities         - Only works for the same machine, then it should be equal
def compare_mesure(params:TestParams, res_in, res_ref):
    # compare data
    compare_pass = 0  # counter for failed tests
    for i_res in res_in.keys():
        i_res_name = i_res.replace(".data", "")
        # check if this data is a sample data or computational data
        try:
            str_last = i_res_name.split("_")[-1]  # extract str between last _ and .data
            if "P" in str_last: raise ValueError()  # go to except statement, as this is a Particle Hash
            is_sample = any(char.isdigit() for char in i_res)  # check if number in data string, then we assume its a sample variable
        except:
            is_sample = False
        sample_grid = 0
        if is_sample: sample_grid = int(i_res_name.split("_")[-1])

        # check if this data is present in reference data
        if not i_res in res_ref.keys():
            params.logError(f"{i_res_name}: Not available for reference results")
            continue
        # check if the data has the same length
        elif len(res_in[i_res]) != len(res_ref[i_res]):
            params.logError(f"{i_res_name}: Different data length. This simulation: {len(res_in[i_res])}, reference: {len(res_ref[i_res])}")
            continue
        # compare values now
        else:
            if i_res.startswith("Hash"):
                if params.same_architecture:
                    diff = [res_in[i_res][i_len] == res_ref[i_res][i_len] for i_len in range(len(res_ref[i_res]))]
                    if sum(diff) == len(res_ref[i_res]):
                        params.logInfo(f'{i_res_name}: Passed')
                    else:
                        if is_sample: params.logError(f'{i_res_name}: Failed hash with matching {diff} for times {res_in[f"Time_s_{sample_grid}.data"]}')
                        else: params.logError(f'{i_res_name}: Failed hash with matching {diff} for times {res_in["Time_s.data"]}')
                        compare_pass += 1
            elif i_res.startswith("Time"):
                diff = res_in[i_res] - res_ref[i_res]
                sum_diff = np.sum(diff)
                if sum_diff != 0:
                    params.logError(f'{i_res_name}: Timings are not equal. This simulation: {res_in[i_res]}, reference: {res_ref[i_res]}')
                    compare_pass += 1
            else:
                diff = np.abs(res_in[i_res] - res_ref[i_res])
                sum_diff = np.sum(diff)
                if sum_diff < params.eps_pass: params.logInfo(f'{i_res_name}: Passed')
                else:
                    if is_sample: params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in[f"Time_s_{sample_grid}.data"]}')
                    else: params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in[f"Time_s.data"]}')
                    compare_pass += 1
    return compare_pass


# function to compare monitoring values:
# Time values               - Mismatches can in theory only occur from different time-stepping settings
# Remapping values          - Check if remapping occurs similar in quantity and at what time
# Inc / Inv error           - if this matches, then it should be a strong hint on a similar simulation as it has to match every single step
def compare_monitoring(params: TestParams, res_in, res_ref):
    # compare data
    compare_pass = 0  # counter for failed tests
    for i_res in res_in.keys():
        i_res_name = i_res.replace(".data", "")
        # check if this data is present in reference data
        if not i_res in res_ref.keys():
            params.logError(f"{i_res_name}: Not available for reference results")
            continue
        # check if the data has the same length
        elif len(res_in[i_res]) != len(res_ref[i_res]):
            params.logError(f"{i_res_name}: Different data length. This simulation: {len(res_in[i_res])}, reference: {len(res_ref[i_res])}")
            continue
        # compare values now
        else:
            if i_res.startswith("Time"):
                if i_res.endswith("c.data"): continue
                diff = res_in[i_res] - res_ref[i_res]
                sum_diff = np.sum(diff)
                if sum_diff != 0:
                    params.logError(f'{i_res_name}: Timings are not equal. This simulation: {res_in["Time_s.data"]}, reference: {res_ref["Time_s.data"]}, diff: {sum_diff}')
                    compare_pass += 1
            elif i_res.startswith("Map"):
                if i_res.endswith("counter.data"):
                    # compute indices where map counter increases
                    map_ind_in = np.where(np.diff(res_in[i_res]))[0] + 1
                    map_ind_ref = np.where(np.diff(res_ref[i_res]))[0] + 1
                    # compare
                    if map_ind_in.size != map_ind_ref.size:
                        params.logError(f'{i_res_name}: Different amount of remappings occurred. This simulation: {map_ind_in.size()}, reference: {map_ind_ref.size()}')
                        compare_pass += 1
                    diff = map_ind_in - map_ind_ref
                    sum_diff = np.sum(diff)
                    if sum_diff != 0:
                        params.logError(f'{i_res_name}: Remapping at different steps. This simulation: {map_ind_in}, reference: {map_ind_ref}')
                        compare_pass += 1
                    else: params.logInfo(f'{i_res_name}: Passed')
            else:  # incompressibility or invertibility error
                diff = np.abs(res_in[i_res] - res_ref[i_res])
                sum_diff = np.sum(diff)
                if sum_diff < params.eps_pass: params.logInfo(f'{i_res_name}: Passed')
                else:
                    params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in["Time_s.data"]}')
                    compare_pass += 1
    return compare_pass


def run_test(params: TestParams):
    # delete old simulation if it exists
    if os.path.isdir(os.path.join(params.root_path, params.wkdir, "data", params.results_loc)):
        shutil.rmtree(os.path.join(params.root_path, params.wkdir, "data", params.results_loc))

    # run simulation
    run_sim(params)

    # load in simulation and reference data
    [res_monitor_in, res_mesure_in] = read_monitoring(params, loc=params.results_loc)
    [res_monitor_ref, res_mesure_ref] = read_monitoring(params, loc=params.reference_loc)

    # compare monitoring and mesure data
    compare_pass = compare_monitoring(params, res_in=res_monitor_in, res_ref=res_monitor_ref)
    compare_pass += compare_mesure(params, res_in=res_mesure_in, res_ref=res_mesure_ref)

    # overwrite reference results
    if (params.save_reference and compare_pass == 0) or params.save_force:
        params.logInfo("Overwriting reference results")
        # rmtree is weird - we have to first remove old folder before copying
        shutil.rmtree(os.path.join(params.root_path, params.wkdir, "data", params.reference_loc))
        shutil.copytree(os.path.join(params.root_path, params.wkdir, "data", params.results_loc), os.path.join(params.root_path, params.wkdir, "data", params.reference_loc))
        # remove particle and time data, as they are currently not needed
        shutil.rmtree(os.path.join(params.root_path, params.wkdir, "data", params.reference_loc, "Particle_data"))
        shutil.rmtree(os.path.join(params.root_path, params.wkdir, "data", params.reference_loc, "Time_data"))
        shutil.rmtree(os.path.join(params.root_path, params.wkdir, "data", params.reference_loc, "Zoom_data"))
    
    # final verdict
    logging.info(f'{params.name} - Completed with {compare_pass} mismatches')

    return compare_pass
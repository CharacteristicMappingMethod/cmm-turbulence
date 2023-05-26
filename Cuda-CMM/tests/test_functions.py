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
    for i_count, i_params in enumerate(params.param_loc):
        params.logInfo(f"Starting simulation {i_count+1} / {len(params.param_loc)}")
        process = Popen(
            [os.path.join(params.root_path, params.exec_loc), 'param_file=' + os.path.join(params.root_path, i_params),
                'workspace=' + os.path.join(params.root_path, params.wkdir) + "/"
            ], stdout=PIPE)
        output, error = process.communicate()
        if not "Finished simulation" in str(output):
            params.logError(f"Simulation {i_count+1} / {len(params.param_loc)} exitted early!")
        params.logInfo(f"Concluded simulation {i_count+1} / {len(params.param_loc)}")


# function to read in monitoring and mesure values
# root_path     - rel path from executed file to main path of cmm
# wkdir         - path of executed test
# loc           - name of data to be loaded
def read_monitoring(params: TestParams, loc):
    params.logInfo(f"Loading monitoring results: {loc}")
    # check if different folder with data exists
    loc_rel = os.path.join(params.root_path, params.wkdir, "data", loc)
    if not os.path.isdir(os.path.join(loc_rel, "Monitoring_data")): params.logError(f"{loc}: No folder 'Monitoring_data'")
    if not os.path.isdir(os.path.join(loc_rel, "Monitoring_data", "Mesure")): params.logError(f"{loc}: No folder 'Mesure'")
    if not os.path.isdir(os.path.join(loc_rel, "Monitoring_data", "Debug_globals")): params.logError(f"{loc}: No folder 'Debug_globals'")
    
    res_return, loc_data = [{}, {}, {}], [os.path.join("data", loc, "Monitoring_data"), os.path.join("data", loc, "Monitoring_data", "Mesure"),
                                          os.path.join("data", loc, "Monitoring_data", "Debug_globals")]
    for i_res, i_loc in zip(res_return, loc_data):
        try:
            for i_load in os.listdir(os.path.join(params.root_path, params.wkdir, i_loc)):
                if i_load.endswith(".data"):
                    if i_load.endswith("Hash.data"):
                        res_read = np.fromfile(os.path.join(params.root_path, params.wkdir, i_loc, i_load), dtype=np.uint8)
                        res_join = ''.join([hex(i_res_in)[2:].upper() for i_res_in in res_read])
                        i_res[i_load] = [res_join[i:i+32] for i in range(0, len(res_join), 32)]
                    else: i_res[i_load] = np.fromfile(os.path.join(params.root_path, params.wkdir, i_loc, i_load), dtype=float)
        except:
            pass
    return res_return


# function to compare mesure values:
# Time values               - Mismatches can in theory only occur from different save settings
# Global quantitities       - Mismatches are compared up to precision of eps_pass
# Hash quantitities         - Only works for the same machine, then it should be equal
def compare_mesure(params:TestParams, res_in, res_ref):
    compare_res = np.array([0, 0])  # counter for passed or failed tests

    # compare keys
    if res_in.keys() != res_ref.keys():
        params.logError(f"Different data available for this simulation and reference results. This simulation: {res_in.keys()}, reference: {res_ref.keys()}")
        compare_res[1] += 1

    # compare data
    for i_res in res_in.keys():
        i_res_name = i_res.replace(".data", "")
        # check if this data is a sample data or computational data
        try:
            sample_grid = int(i_res_name.split("_")[-1])  # if last is only digits then we assume it comes from sample grid size
            is_sample = True
        except:  # elsewise it is not a sampled variable
            is_sample = False
            sample_grid = 0

        # check if this data is present in reference data
        if not i_res in res_ref.keys():
            params.logError(f"{i_res_name}: Not available for reference results")
            compare_res[1] += 1
            continue
        # check if the data has the same length
        elif len(res_in[i_res]) != len(res_ref[i_res]):
            params.logError(f"{i_res_name}: Different data length. This simulation: {len(res_in[i_res])}, reference: {len(res_ref[i_res])}")
            compare_res[1] += 1
            continue
        # compare values now
        else:
            if i_res.startswith("Hash"):
                if params.same_architecture:
                    diff = [res_in[i_res][i_len] == res_ref[i_res][i_len] for i_len in range(len(res_ref[i_res]))]
                    if sum(diff) == len(res_ref[i_res]):
                        params.logInfo(f'{i_res_name}: Passed')
                        compare_res[0] += 1
                    else:
                        if is_sample: params.logError(f'{i_res_name}: Failed hash with matching {diff} for times {res_in[f"Time_s_{sample_grid}.data"]}')
                        else: params.logError(f'{i_res_name}: Failed hash with matching {diff} for times {res_in["Time_s.data"]}')
                        compare_res[1] += 1
            elif i_res.startswith("Time"):
                diff = res_in[i_res] - res_ref[i_res]
                sum_diff = np.sum(diff)
                if sum_diff != 0:
                    params.logError(f'{i_res_name}: Timings are not equal. This simulation: {res_in[i_res]}, reference: {res_ref[i_res]}')
                    compare_res[1] += 1
            else:
                diff = np.abs(res_in[i_res] - res_ref[i_res])
                sum_diff = np.sum(diff)
                if sum_diff < params.eps_pass:
                    params.logInfo(f'{i_res_name}: Passed')
                    compare_res[0] += 1
                else:
                    if is_sample: params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in[f"Time_s_{sample_grid}.data"]}')
                    else: params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in[f"Time_s.data"]}')
                    compare_res[1] += 1
    return compare_res


# function to compare debug values of arrays:
# Global quantitities       - Mismatches are compared up to precision of eps_pass
# Hash quantitities         - Only works for the same machine, then it should be equal
def compare_debug(params: TestParams, res_in, res_ref):
    compare_res = np.array([0, 0])  # counter for passed or failed tests

    # compare keys
    if res_in.keys() != res_ref.keys():
        params.logError(f"Different data available for this simulation and reference results. This simulation: {res_in.keys()}, reference: {res_ref.keys()}")
        compare_res[1] += 1

    # compare data
    for i_res in res_in.keys():
        i_res_name = i_res.replace(".data", "")
        # check if this data is a sample data or computational data
        is_zoom = "Zoom" in i_res_name
        if is_zoom: zoom_ctr = int(i_res_name.split("_")[-4 + 3*("Time" in i_res_name)])
        else: zoom_ctr = 0
        try:
            sample_grid = int(i_res_name.split("_")[-2 + 4*is_zoom])  # if second last is only digits then we assume it comes from sample grid size
            is_sample = True
        except:  # elsewise it is not a sampled variable
            is_sample = False
            sample_grid = 0

        # check if this data is present in reference data
        if not i_res in res_ref.keys():
            params.logError(f"{i_res_name}: Not available for reference results")
            compare_res[1] += 1
            continue
        # check if the data has the same length
        elif len(res_in[i_res]) != len(res_ref[i_res]):
            params.logError(f"{i_res_name}: Different data length. This simulation: {len(res_in[i_res])}, reference: {len(res_ref[i_res])}")
            compare_res[1] += 1
            continue
        # compare values now
        else:
            if i_res_name.endswith("Hash"):
                if params.same_architecture:
                    diff = [res_in[i_res][i_len] == res_ref[i_res][i_len] for i_len in range(len(res_ref[i_res]))]
                    if sum(diff) == len(res_ref[i_res]):
                        params.logInfo(f'{i_res_name}: Passed')
                        compare_res[0] += 1
                    else:
                        time_name = "Time_s" + is_sample*f"_{sample_grid}" + is_zoom*f"_Zoom_{zoom_ctr}" + ".data"
                        params.logError(f'{i_res_name}: Failed hash with matching {diff} for times {res_in[time_name]}')
                        compare_res[1] += 1
            elif i_res.startswith("Time"):
                diff = res_in[i_res] - res_ref[i_res]
                sum_diff = np.sum(diff)
                if sum_diff != 0:
                    params.logError(f'{i_res_name}: Timings are not equal. This simulation: {res_in[i_res]}, reference: {res_ref[i_res]}')
                    compare_res[1] += 1
            else:
                diff = np.abs(res_in[i_res] - res_ref[i_res])
                sum_diff = np.sum(diff)
                if sum_diff < params.eps_pass:
                    params.logInfo(f'{i_res_name}: Passed')
                    compare_res[0] += 1
                else:
                    time_name = "Time_s" + is_sample*f"_{sample_grid}" + is_zoom*f"_Zoom_{zoom_ctr}" + ".data"
                    params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in[time_name]}')
                    compare_res[1] += 1
    return compare_res


# function to compare monitoring values:
# Time values               - Mismatches can in theory only occur from different time-stepping settings
# Remapping values          - Check if remapping occurs similar in quantity and at what time
# Inc / Inv error           - if this matches, then it should be a strong hint on a similar simulation as it has to match every single step
def compare_monitoring(params: TestParams, res_in, res_ref):
    compare_res = np.array([0, 0])  # counter for passed or failed tests

    # compare keys
    if res_in.keys() != res_ref.keys():
        params.logError(f"Different data available for this simulation and reference results. This simulation: {res_in.keys()}, reference: {res_ref.keys()}")
        compare_res[1] += 1

    # compare data
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
                    compare_res[1] += 1
            elif i_res.startswith("Map"):
                if i_res.endswith("counter.data"):
                    # compute indices where map counter increases
                    map_ind_in = np.where(np.diff(res_in[i_res]))[0] + 1
                    map_ind_ref = np.where(np.diff(res_ref[i_res]))[0] + 1
                    # compare
                    if map_ind_in.size != map_ind_ref.size:
                        params.logError(f'{i_res_name}: Different amount of remappings occurred. This simulation: {map_ind_in.size()}, reference: {map_ind_ref.size()}')
                        compare_res[1] += 1
                        continue
                    diff = map_ind_in - map_ind_ref
                    sum_diff = np.sum(diff)
                    if sum_diff != 0:
                        params.logError(f'{i_res_name}: Remapping at different steps. This simulation: {map_ind_in}, reference: {map_ind_ref}')
                        compare_res[1] += 1
                    else:
                        params.logInfo(f'{i_res_name}: Passed')
                        compare_res[0] += 1
            else:  # incompressibility or invertibility error
                diff = np.abs(res_in[i_res] - res_ref[i_res])
                sum_diff = np.sum(diff)
                if sum_diff < params.eps_pass:
                    params.logInfo(f'{i_res_name}: Passed')
                    compare_res[0] += 1
                else:
                    params.logError(f'{i_res_name}: Failed with difference {diff} for times {res_in["Time_s.data"]}')
                    compare_res[1] += 1
    return compare_res


def run_test(params: TestParams):
    # delete old simulation if it exists
    for i_res in params.results_loc:
        if os.path.isdir(os.path.join(params.root_path, params.wkdir, "data", i_res)):
            shutil.rmtree(os.path.join(params.root_path, params.wkdir, "data", i_res))

    # run simulation
    run_sim(params)

    # load in simulation (from last simulation in case of several) and reference data
    [res_monitor_in, res_mesure_in, res_debug_in] = read_monitoring(params, loc=params.results_loc[-1])
    [res_monitor_ref, res_mesure_ref, res_debug_ref] = read_monitoring(params, loc=params.reference_loc)

    # compare monitoring and mesure data
    compare_res = compare_monitoring(params, res_in=res_monitor_in, res_ref=res_monitor_ref)
    compare_res += compare_mesure(params, res_in=res_mesure_in, res_ref=res_mesure_ref)
    compare_res += compare_debug(params, res_in=res_debug_in, res_ref=res_debug_ref)

    # remove heavy data, as they are currently not needed
    for i_res in params.results_loc:
        res_ref = os.path.join(params.root_path, params.wkdir, "data", i_res)
        if os.path.isdir(os.path.join(res_ref, "Particle_data")): shutil.rmtree(os.path.join(res_ref, "Particle_data"))
        if os.path.isdir(os.path.join(res_ref, "Time_data")): shutil.rmtree(os.path.join(res_ref, "Time_data"))
        if os.path.isdir(os.path.join(res_ref, "Zoom_data")): shutil.rmtree(os.path.join(res_ref, "Zoom_data"))
        if os.path.isdir(os.path.join(res_ref, "MapStack")): shutil.rmtree(os.path.join(res_ref, "MapStack"))

    # overwrite reference results
    if (params.save_reference and compare_res[1] == 0) or params.save_force:
        params.logInfo("Overwriting reference results")
        # rmtree is weird - we have to first remove old folder before copying
        rel_ref = os.path.join(params.root_path, params.wkdir, "data", params.reference_loc)
        if os.path.isdir(rel_ref): shutil.rmtree(rel_ref)
        shutil.copytree(os.path.join(params.root_path, params.wkdir, "data", params.results_loc[-1]), rel_ref)
    
    # final verdict
    logging.info(f'{params.name} - Completed with {compare_res[1]} mismatches and {compare_res[0]} passed comparisons')

    return compare_res
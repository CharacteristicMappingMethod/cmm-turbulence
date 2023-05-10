This readme files explains the example files. The first form an introduction into how to use and understand all parameters,
while the later starting from example 10 build reference simulations for specific initial conditions.




*****   Example 01 - Working example   *****
This example is meant to test, that the code is working on your machine.
In addition, it gives a major overview on the file structure and log output itself.
Make sure to check the preparations before running the code in order to ensure, that everything was setup correctly.

Run the code inside /Cuda-CMM:
./SimulationCuda2d.out param_file=./examples/euler2D/params-01-working-test.txt

Expected output:
A new file will be created in the data-folder named '01-working-test_two_cosine_C32_F32_t8_T1'.
Here, the addition of '_two_cosine_C32_F32_t8_T1' gives further details of the settings of the simulation:

Initial condition	-	two_cosine
Coarse grid			-	32x32
Fine grid			-	32x32
Step-size			-   1/8
Final time			-	1

Further simulation settings can be taken from the parameter file.
The actual used parameters can be found inside the result folder in 'params.txt'
Also, the log output to the console is logged to the file 'log.txt'.
	It should contain some details of the simulation, ending with the line 'Memory initialization finished'
	Afterwards, the simulation initialization starts, which should do 6 initialization steps in total,
	ending with the line 'Simulation initialization finished'.
	Thereafter the actual simulation loops starts, which computes 8 time-steps and then finishes the simulation.
	The last line should conclude with 'Finished simulation		Last Cuda Error = cudaSuccess'
	The overall computation time (c-time) on the test machine took 0.2s.
Two folders will be created in the result folder, being 'Time_data' and 'Monitoring_data'
The folder 'Time_data' should be empty, as no variables were set to be saved
The folder 'Monitoring_data' contains monitored values over time:
	'Time_s'					-	Time instants of each time-step, can be non-uniform to match specific time targets
	'Time_c'					-	Real time in seconds from begin of simulation
	'Error_incompressibility'	-	Incompressibility error, also shown in log-file
	'Map_counter'				-	Map counter, increasing for each sub-map
	'Map_gaps'					-	Distance to last remapping
The sub-folder 'Mesure' inside 'Monitoring_data' contains all monitored global quantities over time. Those files are empty for this run.
	'Time_s'		-	Time instants of each mesurement
	'Energy'		-	Energy measured in L2 norm over the whole domain, computed from stream function Hermite entries
	'Enstrophy' 	-	Enstrophy measured in L2 norm over the whole domain, computed from fine vorticity of sub-map initial condition
	'Max_vorticity'	-	Maximum of vorticity, of course
	'Palinstrophy'	-	Palinstrophy measured in L2 norm over the whole domain, computed from vorticity with spectral derivatives



*****   Example 02 - First simulation   *****
This example should be a first simulation with results. It aims to explain all basic parameters of the parameter-file.

Run the code inside /Cuda-CMM:
./SimulationCuda2d.out param_file=./examples/euler2D/params-02-first-simulation.txt

Expected output:
A new file will be created in the data-folder named '02-first-simulation_4_nodes_C128_F128_t64_T1'.
The log-file shows 64 computed time-step. It also gives information on other things occuring
	During the run, conservation was computed 4 times, which can be seen by the 4 lines containing 'Coarse Cons'.
	Two remappings occured, one at step 28 and one at step 62
	The overall computation time (c-time) on the test machine took 1s.
The parameter file set several important factors for the computations. Those will be explained here:
	workspace					-	location of current workspace, data will be saved in data-sub-folder
	sim_name					-	distinct name of the simulation
	grid_coarse					-	size of coarse grid, estimated by balance of CPU RAM memory available for remappings and accuracy
	grid_fine					-	size of fine grid, estimated by balance of computation speed for remappings and GPU RAM memory available
	grid_psi					-	size of psi grid, should be best set as equal or slightly higher than coarse grid
	grid_vort					-	size of vortictiy grid for stream function evaluaten, should be set after fine grid
	final_time					-	final time to which the simulation advances
	set_dt_by_steps				-	bool, if step-size will be set after steps or relative to grid
	steps_per_sec				-	how many steps per simulation time
	save_computational_num		-	amount of save settings for computationals, explained later
	mem_RAM_CPU_remaps			-	CPU memory for remapping in MB
	save_map_stack				-	bool, if map stack should be saved
	verbose						-	level of output to console or log-file, going from 0 to 3
	initial_condition			-	name of initial condition
	initial_params				-	Settings of parameters for initial conditions
	initial_discrete			-	bool, if initial condition is set after discrete data
	incomp_threshold			-	incompressibility threshold for remapping condition
	map_epsilon					-	map epsilon for GALS stencil, should be around 1-5e-4
	time_integration			-	name of time integration method
	lagrange_override			-	set lagrange order explicitly, should be done after time integration methods
	lagrange_init_higher_order	-	bool, for how velocity should be initialized, is usually not changed
	map_update_order			-	order for map update stencil, is usually not changed
	molly_stencil				-	include mollification, is usually not changed
	freq_cut_psi				-	frequency for low-pass filter, important for reducing number of remappings
	skip_remapping				-	bool, skip remapping procedure, usefull for validation test
	save_sample_num				-	amount of save settings for samples, explained later
	scalar_name					-	name of scalar initial condition
	scalar_discrete				-	should scalar initial condition be set after discrete data
	save_zoom_num				-	amount of save settings for zooms, explained later
	forward_map					-	bool, should forward map be computed
	particles_advected_num		-	amount of particle sets embedded in the flow, explained later
The Mesure values in 'Monitoring_data/Mesure' now contain 4 data points each, which can be taken from the file size of 32 byte
Reading in those data for postprocessing to other programs is fairly easy, as the data is only encoded in binary data
	In python, this can be read in easily using np.fromfile(location)
The simulation also saved data at two time instants, which can be seen in the 'Time_data'-folder
	Each saved time instant will have its one sub-folder for data in the form of 'Time_{T}'
	Only exception: Time_final, this should usually be the final time, but can also be earlier in case the simulation quitted earlier
	In the parameter file, Time_final is always referenced as 10000
For this simulation, the vorticity and velocity were saved at the initial and final time
	They are also encoded in binary data, their grid information is hinted at the end of the file names



*****   Example 03 - Different save settings   *****
This example aims to explain the different options to save and introduces the zoom property.

Run the code inside /Cuda-CMM:
./SimulationCuda2d.out param_file=./examples/euler2D/params-03-save-settings.txt

Expected output:
A new file will be created in the data-folder named '03-save_settings_4_nodes_C128_F128_t64_T4'.
The log-file shows 86 remappings with remapping occuring every step at the end, hinting at a badly converged solution
	Increasing the map size, the incompressibility treshold or decreasing the lowpass filter width should help to stabilize the solution
	The overall computation time (c-time) on the test machine took 7s.
	Conservation for computational (coarse) and sampled variables can be found scattered in the log-file
The parameter-file shows several settings, which were made in order to save variables of the simulation.
save_computational controls the saving of variables used for computations
	is_instant	-	bool, if the saving is done in an interval or only at one time instant
	time_start	-	first time instant to be saved (is_instant=0) or only time instant (is_istant=1)
	time_end	-	final time at which the variables could be saved, only for time_instant=0
	time_step	-	in what frequency should the variables be saved, only for time_instant=0
	var			-	which variables should be saved? an up-to-date list can be found in the settings.cu-file
	conv		-	bool, should conservation be computed or not
The different sets used here should showcase several interesting use-cases for every simulation
	1st set -	used to save the initial vorticity and velocity in order to check it later
	2nd set	-	used to save the final vorticity, internally 10000 is treated as the 'final' time
	3rd set	-	used to consistently compute the convergence, as no variables are saved, this is only used for convergence
	4th set	-	used to showcase continous saving of variables
save_sample controls saving of sampled variables, where the whole map-stack was applied, those values are usually used for post-processing
	grid		-	on what uniform grid should the values be computed
	all other values are similar to save_computational, except of a missing conv-option, as convergence will always be computed
	different sets can have different gridsizes
	the second set showcases, how time instants can be set even when they are not on the time-instants, as the computation will adapt to it
	each different grid will always compute and save conservation properties and saves it independently
save_zoom controls computation of zooms, again utilizing the whole map-stack
	pos_x	-	center point of zoom window in x-direction
	pos_y	-	center point of zoom window in y-direction
	width_x	-	width of zoom window in x-direction
	width_y	-	width of zoom window in y-direction
	rep		-	how many zooms should be computed with in- or decreasing window size, 1 for no further zooms
	rep_fac	-	factor of window size change for each step
Computational and sampled saves can be found in 'Time_data', while zooms are located in 'Zoom_data'.



*****   Example 04 - Computation containing particle sets   *****
This example aims to explain the different settings in order to embedd fluid and inertial particles.

Run the code inside /Cuda-CMM:
./SimulationCuda2d.out param_file=./examples/euler2D/params-04-embedded-particles.txt

Expected output:
A new file will be created in the data-folder named '04-embedded_particles_4_nodes_C128_F128_t64_T4'.
The log-file shows the inclusion of 3 particle sets at the beginning in the initialization.
	The overall computation time (c-time) on the test machine took 6s.
The settings for the particle sets can be found in the parameter file within the lines starting with 'particles_advected'
	The first set features a set of randomly distributed fluid particles in a rectangle in the center
	The second set features a set of randomly distributed particles with Stokes number estimated as 1 over the whole domain
	The third set features a set of heavy particles distributed in a ring around the center, embedded at t=1
The different parameters describe the behaviour of the particle sets
	num					-	amount of particles
	tau					-	particle relaxation time, 0 for fluid particles, 0.5 for St=1
	seed				-	random seed for random initial positions
	time_integration	-	abbreviation for time integration scheme used, recommended: RK3Mod or RK4Mod
	init_name			-	name of initial condition, see init.cu
	init_time			-	time at which particles will be embedded in flow
	init_vel			-	bool, will initial velocity for inertial particles be set after the flow (1) or to 0 (0)
	init_param_1		-	individual parameter for initial conditions, see init.cu, mainly center_x
	init_param_2		-	individual parameter for initial conditions, see init.cu, mainly center_y
	init_param_3		-	individual parameter for initial conditions, see init.cu, mainly width_x
	init_param_4		-	individual parameter for initial conditions, see init.cu, mainly width_y
The individual particle sets can be saved in the save_computational options
	using the form PartA_XX, with XX being the number of the particle set starting from 01 to save
	saving option saves in new folder 'Particle_data', with sub-folders similar to 'Time_data'
	all individual particle sets are differentiated by their ID, allowing up to 99 particle sets to be used per simulation



*****   Example 05 - Forward map computation   *****
This example aims to explain a computation utilizing the forward map in order to advect particle positions.

Run the code inside /Cuda-CMM:
./SimulationCuda2d.out param_file=./examples/euler2D/params-05-forward-map.txt

Expected output:
A new file will be created in the data-folder named '04-embedded_particles_4_nodes_C128_F128_t64_T4'.
The log-file shows the inclusion of 3 particle sets at the beginning in the initialization.
	Two new error computations are computed: incompressibility error for forward map and invertibility error
	The second particles are embedded at t=0.1, where a step and remapping can be observed
	The overall computation time (c-time) on the test machine took 10.5s.
Similar to advected particles, fluid particles can be forwarded using the forward map, those sets are initialized quite similar
	They are set under 'particles_forwarded'
	The parameters are similar to particles, with only 'tau', 'time_integration' and 'init_vel' missing
The save settings can be set in save_sample or save_zoom for forwarded particles
	using the form PartF_XX, with XX being the number of the particle set starting from 01 to save
	particles are computed using the whole map stack of forward maps
Forwarded maps have the same size than the backwards map (coarse grid) and are remapped at the same time








*****   Example 10 - Reference simulation for all validations compared to Bruce's simulations   *****
This example was used as the reference case in order to compare to the results of Yin et al. (2020).
It features the main settings with the computation of the 4mode flow until t=1 with increased psi-grid.
No remappings or filters were used during the simulation.
The simulation takes up approximately 1GB of GPU RAM and 4GB of CPU RAM.
Conservation is computed at beginning and ending, in addition the backwards map and vorticity is saved at the final time.



*****   Example 11 - Reference simulation suitable for investigation of remapping procedure   *****
This example was used as the reference case for all hyperparameter investigations.
It features the main settings with the computation of the 4mode flow until t=4 with larger fine grid.
Remappings are enabled and a reference low-pass filter of 128 was used.
The simulation takes up approximately 1GB of GPU RAM and 4GB of CPU RAM.
Conservation is computed at beginning and every s-time, in addition the backwards map and vorticity is saved every s-time as well.

Example 11b is the same but with coarser settings, this is for development tests to ensure corect outputs.


*****   Example 12 - V100 run of shear layer flow   *****
An example of a shear layer flow to be run on V100. This features the computation of a developing shear layer flow.
Usage on V100 in resource folder: make all; nohub ./SimulationCuda2d.out param_file=[param-file-location]
It saves a zoom between the two emerging vortices every s-time until t=25, vortices would merge starting from t=30.
Log features 206 maps and a final time of almost 9h of computation time.



*****   Example 13 - V100 run of ring in isotropic turbulence using 'gaussian_blobs'   *****
An example of emerging isotropic turbulence from a checkerboard-grid of gaussian vortices with small random displacement.
Usage on V100 in resource folder: make all; nohub ./SimulationCuda2d.out param_file=[param-file-location]
It saves particle positions of particles distributed in a ring starting from t=10 at every 0.5 s-time until t=30 for 4 particle sets.
It additionally saves Vorticity and Velocity at 4096-grid every 5 s-time.
Log featured 663 maps and a final time of roughly 25.5h of computation time.
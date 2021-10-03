"""A program to generate detailed maps for spherical worlds."""

import os
import sys
import argparse
import time
# import math
import json
import concurrent.futures
# import meshio as mi
import meshzoo as mz
import numpy as np
from numba import njit, prange
import pyvista as pv
from scipy.spatial import KDTree
import opensimplex as osi
from util import *
from erosion import *
from climate import *
# pylint: disable=not-an-iterable

# ToDo List:
# - In the future when we have climate simulation, run a test to see if it's faster to lookup xyz2latlon as-needed or to simply store latlon for each vertex in a big master array.
#    Or instead of storing it in RAM--and this is something applicable to all arrays--to limit RAM footprint it might be better to store most arrays in sqlite and read values
#    from disk only as-needed. However, this would depend on read/write speeds, which will vary drastically across user hardware from m.2 to 5400 RPM drives. Also I don't want to kill user drives with GBs of writes.
# - See if there is any sort of pattern in the vertex order from meshzoo that would allow a LOD-like system or a way to know every other vertex or every N vertices so that
#    we could use a sparser or lower-fidelity (low-fi) version of the dense sphere for some calculations. (e.g. I'm thinking of collision detection between drifting continents.)
#    One possibly-horrifying idea would be to generate two different spheres with meshzoo at different subdivision levels, build a KD-Tree for the lower-subdivision mesh,
#    do the low-fi calculations on that one, and then we query the low-fi's KD-Tree for each vert XYZ from the *high-res* mesh for the nearest low-fi vert to apply the result *to* the high-res.
#
# - More powerful performance profiling/benchmarking with cprofile
#    https://docs.python.org/3/library/profile.html
#    https://www.machinelearningplus.com/python/cprofile-how-to-profile-your-python-code
#    Or maybe timeit https://docs.python.org/3/library/timeit.html
#    https://www.geeksforgeeks.org/timeit-python-examples
# - In addition to speed profiling, also do some memory profiling because this could end up needing many GB of RAM.
# - Note: Voroni tesselation library: https://github.com/gdmcbain/voropy
# - Idea: Meshzoo is already really fast; I wonder if it can be @njitted (Answer: Not easily)
# - One thing to test is that it may be faster to use predefined vertex, triangle, and adjacency arrays that we read from disk instead of building the mesh with meshzoo and building the adjacency every single time.
#    This would mean there is less flexibility in choosing what level of subdivision to use when creating worlds but it might save on computation time and possibly on RAM since we don't have to make various copies
#    of arrays to build adjacency and such.  Maybe not so useful while I'm building the program but it could be useful once the program is 'production ready'.
#    I could optionally ship the program with precomputed arrays for download or include a precompute_arrays() function so the user can do it themselves (depending on how large the arrays are.. they'd probably be huge on disk)
#    so they only need to precompute once.
# - Find out which takes up more bytes in RAM: color represented as RGB, HSV, or hex value, and if the 'winner' at a bit depth of 8 is still the winner at 16 or higher.
# - All important variables for things like tectonics, erosion, and climate need to be 'ratios' that are resolution and radius-independent.  So they will all be related to the distance between vertices, which is affected by radius and subdivisions.
# - Far future idea: This would practically be it's own implementation, but it might be possible to take only a smaller square/rect slice of a given planet to export at a higher resolution.
#    Use the latlon2xyz to make a grid from a min/max lat and lon, then build the triangulated connectivity basically the same way Sebastian Lague does it. With enough octaves the opensimplex will have detail down to that level.
#    Things like terrain resources (metal deposits, stone type, etc.) and temperature/precipitation have pretty slow interpolations at planet scale so these could just be a nearest neighbor search from a KD-Tree of the main icosphere.
#    Erosion would be the hard part, because the grid mesh would have distorted area, wouldn't it? That's the whole problem with lat/lon representations of a sphere.  Unless we grab a proper curved slice from 3D where the sides bow
#    and aren't perfect vertical/horizontal lat/lon lines.  A lower-resolution erosion from the icosphere could be KD-Tree'd onto the new grid as a starting point for elevations, interpolated, and THEN sample opensimplex at higher resolution?
#    Using point sampling on the new grid in an uneven fashion (uneven in proportion to lat/lon, so more dense at the bottom, less dense at the top) could maybe produce an "even" representation for an erosion graph.
#    An alternative might be to take a lat/lon slice of the meshzoo icosasphere of some N division level, then subdivide the edges to make new triangles are higher resolution and use those for the same type of erosion graph that we're already doing.
# - Fun, but distant, idea: Calculate zones that are likely to have high presence of fossils. In theory this should be doable once I'm tracking the types of rock in the ground, the sediment, the water distribution, and plant growth as a proxy for life/death.
# - Utilize Logger and debug levels, and pray numba won't hate it. https://docs.python.org/3/library/logging.html
#    https://www.reddit.com/r/learnpython/comments/ph9hp4/how_tiring_is_print_function_for_the_computer/ (useful tip: can use modulus to only record every X iterations, e.g. for i in range whatever, if i % 100 == 0: print(stuff))
#    https://stackoverflow.com/questions/60077079/how-can-i-disable-numba-debug-logging-when-debug-env-variable-is-set
# - Maybe some fancy progress bars. https://tqdm.github.io/

def main():
    """Main function."""

# Initial setup stuff
# =============================================
    parser = argparse.ArgumentParser(
        description="Generate maps for spherical worlds.")
    parser.add_argument("-n", "--name", type=str, 
        help="World name (without file extension). If not specified then \
        a default name will be used.")
    parser.add_argument("-d", "--divisions", type=int, default=320, 
        help="Number of divisions to add to the planet mesh. 320 will make a \
        mesh with ~2 million triangles. 2500 makes a 125mil triangle mesh.")
    parser.add_argument("-s", "--seed", type=int, 
        help="A number used to seed the RNG. If not specified then a random \
        number will be used.")
    parser.add_argument("-r", "--radius", type=float, 
        help="Planet radius in meters. Optional. Default is 1 meter with all \
        attributes scaled to fit.")
    parser.add_argument("-t", "--tilt", type=float, default=0.0, 
        help="Axial tilt of the planet, in degrees.")
    # parser.add_argument("-lc", "--loadconfig", type=str, 
    #     help="Load the generation settings from a text file. NOTE: Any other \
    #     provided arguments will override individual settings from the file.")
    parser.add_argument("--mesh", action="store_true", 
        help="Export the world as a 3D mesh file.")
    # parser.add_argument("--pointcloud", action="store_true", 
    #     help="Export the world as a 3D point cloud file.")
    # parser.add_argument("--database", action="store_true", 
    #     help="Export the world as a sqlite file.")
    parser.add_argument("--png", action="store_true", 
        help="Save a png map of the world.")
    parser.add_argument("--config", action="store_true", 
        help="Save the generation settings for a given world as a text file.")

    args = parser.parse_args()

    if args.divisions <= 0:
        print("\n" + "Divisions must be an even, positive integer." + "\n")
        sys.exit(0)
    if args.divisions > 2250:
        print("\n" + "WARNING. Setting divisions to a large value can use gigabytes of RAM.")
        print("         The 3D mesh will have (divisions * divisions * 10 + 2) vertices.")
        print("         The 3D mesh will have (divisions * divisions * 20) triangles.")
        print("         Simulation will also take proportionately longer." + "\n")
        confirm = input("Continue anyway? Y/N: ")
        if confirm.lower() not in ('y', 'yes'):
            sys.exit(0)
    if (args.divisions % 2) == 0:
        divisions = args.divisions
    else:
        divisions = args.divisions + 1
        print(f"Even numbers only for divisions, please. Setting divisions to {divisions}.")

    if args.seed:
        world_seed = args.seed
    else:
        seed_rng = np.random.default_rng()
        world_seed = int(seed_rng.integers(low=0, high=999999))
        # world_seed = int(seed_rng.integers(low=-999999, high=999999))
        # rng = np.random.default_rng(i)
        # rng.random()

    if args.name:
        world_name = args.name  # ToDo: Test for delicacy and safety with various inputs such as forward/backward slashes, 'invalid' weird inputs, symbols, etc. (slashes result in a 'safe' file path error)
    else:
        world_name = str(world_seed) + "_" + str(divisions)

    if args.radius:  # ToDo: I wonder if it's proper to do like "world_radius = args.radius or 1.0" to save a few lines of code
        world_radius = abs(args.radius)
    else:  # NOTE: Technically speaking I should do the same as I did for divisions and simply set a default in argparse and not have this if/else statement at all.
        # world_radius = 6378100.0  # Actual earth radius in meters
        world_radius = 1.0  # In meters

    if -90.0 <= args.tilt <= 90.0:  # ToDo: Make sure that the temperature and daily rotation calculations aren't fragile and then expand this to allow larger tilts.
        axial_tilt = args.tilt      # e.g. Uranus and Venus have tilts > 90 (97 and 177, respectively) and several planets have retrograde rotation.
    else:
        print("\n" + "Axial tilt must be in the range -90 to +90." + "\n")
        sys.exit(0)

    world_albedo = 0.31
    orbital_distance = 149597870.7  # 1 AU in km
    star_radius = 696340
    star_temp = 5778  # Kelvin

    my_dir = os.path.dirname(os.path.abspath(__file__))
    options = load_settings("options.json")
    img_width = options["img_width"]
    img_height = options["img_height"]
    save_dir = os.path.join(my_dir, options["save_folder"])
    export_list = options["export_list"]

    KDT = None
    test_latlon = False

    do_erode = False
    do_climate = True

    snapshot_erosion = False
    snapshot_climate = False

    # ToDo: We should possibly only try making the dir at the moment we save.
    try:  # ToDo: Test if the directory already exists. Maybe even attempt to see if we have write permission beforehand.
        os.mkdir(save_dir)
    except:
        # ToDo: Actual exception types
        # print("Failed to create script output directory! Check folder permissions.")
        pass

    world_config = {
        "world_name": world_name,
        "world_seed": world_seed,
        "divisions": divisions,
        "world_radius": world_radius,
        "axial_tilt": axial_tilt,
        "world_albedo": world_albedo,
        "orbital_distance": orbital_distance,
        "star_radius": star_radius,
        "star_temp": star_temp
    }

    # Save the world configuration as a preset
    if args.config:
        save_settings(world_config, save_dir, world_name + '_config', fmt=options["settings_format"])

# Start the party
# =============================================
    runtime_start = time.perf_counter()
    now = time.localtime()
    print("==============================")
    print("World seed:", world_seed)
    print(f"Script started at: {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}")

    points, cells = create_mesh(divisions)
    if world_radius != 1:
        points *= world_radius

    # print("Points:")
    # print(points)
    # print("Cells:")
    # print(cells)

    # If I need to output several meshes for examination
    # for i in range(2, 26, 2):
    #     a, b = create_mesh(i)
    #     save_mesh(a, b, save_dir, f"{i:02}" + '_smooth')

# Saving the mesh here just writes a plain sphere with no elevation offsets (for debug)
    if args.mesh:
        save_mesh(points, cells, save_dir, world_name + '_smooth')

# Prepare KD Tree and stuff for image saving
# =============================================

    # ToDo: Test compact_nodes and balanced_tree args for build/query performance tradeoffs
    if args.png or snapshot_erosion or snapshot_climate:
        time_start = time.perf_counter()
        ll = make_ll_arr(img_width, img_height)
        ll_end = time.perf_counter()
        KDT = build_KDTree(points)
        kdt_end = time.perf_counter()
        img_query_data = KDT.query(ll, k=3, workers=-1)
        query_end = time.perf_counter()

        ll = None
        KDT = None

        print(f"  LL built in       {ll_end - time_start :.5f} sec")
        print(f"  KD Tree built in  {kdt_end - ll_end :.5f} sec")
        print(f"  Query finished in {query_end - kdt_end :.5f} sec")


# Sample noise
# =============================================
    # Initialize the permutation arrays to be used in noise generation
    time_start = time.perf_counter()
    perm, pgi = osi.init(world_seed)
    time_end = time.perf_counter()
    print(f"Time to init permutation arrays: {time_end - time_start :.5f} sec")

    # Ocean altitude is a *relative* percent (expressed as a decimal) of the max altitude
    # e.g. 0.4 would set the ocean to 40% of the height from the min to max altitude.
    ocean_percent = 0.55
    n_init_rough = 1.5     # Initial roughness of first octave
    n_init_strength = 0.4  # Initial strength of first octave
    n_roughness = 2.5      # Multiply roughness by this much per octave
    n_persistence = 0.5    # Multiply strength by this much per octave
    n_octaves = 7          # Number of octaves
    # ToDo: Figure out elevation scaling.  Earth's avg radius is 6.3781 million meters (6378 km). We consider "0" to be sea level. So I need to:
    # 1. Get my heights in range -1 to +1 ? (This might not be required but might make step 3/4 more annoying than working with 0? Or not; the rescale process itself can reset the sea level to be the 0 altitude.)
    # 2. then choose sea level as a percent between the min and max
    # 3. For only heights above that sea level, rescale so that the max altitude is something reasonable (taking into account that a max height in 'absolute' m or km has to be scaled down to the sphere radius)
    # 4. Do the same for heights below that sea level and some determined min value
    # For reference, Mt. Everest is ~8848.86 m above sea level, the lowest land is the Dead Sea at -428 m below sea level, and the Mariana Trench has a depth of at least -10984 m below sea level. "Average" ocean floor is ~3.8 km.
    print("Sampling noise...")

    height = sample_octaves(points, None, perm, pgi, n_octaves, n_init_rough, n_init_strength, n_roughness, n_persistence, ocean_percent, world_radius)  # ToDo: Consider pulling these values from seed json file cuz this line is a long boi now.

    height = rescale(height, -0.05, 0.15)
#    height = rescale(height, -4000, 8850)
#    height = rescale(height, -4000, 0, 0, mode='lower')
#    height = rescale(height, 0, 8850, 0, mode='upper')

    minval = np.amin(height)
    maxval = np.amax(height)
    # https://math.stackexchange.com/questions/2110160/find-percentage-value-between-2-numbers
    # ocean_level = minval + ((maxval - minval) * 50 / 100)  # If I want to input percentage as an integer between 0 and 100 we need the divide by 100 step
    ocean_level = minval + ((maxval - minval) * ocean_percent)  # Or this way to just input a decimal percent. NOTE: Due to the way np.clip works, values less than 0% or greater than 100% have no effect.
    print("  Ocean Level:", ocean_level)

    # height *= 10
    # height += 6378100
    # height += 400000
    # height += world_radius - 1

    height = np.clip(height, ocean_level, maxval)  # Hack for showing an ocean level in pyvista

    print("  min:", minval)
    print("  max:", maxval)

    if args.png and not do_erode:
        export_list["height"] = height

# Erode! 
# ToDo: Which came first, the erosion or the climate? Maybe do climate first, as the initial precipitation determines erosion rates.
#   Temperature also determines evaporation rate which affects how much erosion we can do.
#   And if we're being fancy with Thermal Erosion instead of approximating then we need surface temps first.
#   However then the altitude-based temperature will be wrong, so, climate has to run again after erosion. (Really they probably need to alternate)
# =============================================
    if do_erode:
        neighbors = build_adjacency(cells)
        sort_adjacency(neighbors)

        # print(height)
        # print(type(height[0][0]))
        # ToDo: Should probably have an ocean clip mask that I can pass through for
        # verts that should be ignored during erosion, although if land borders
        # ocean then it should be able to add to the ocean floor to some degree
        # until uplifts above the ocean level, at which point it is removed from
        # the mask and becomes part of the land points.

        erode_start = time.perf_counter()
        if snapshot_erosion:
            erode_terrain3(points, neighbors, height, num_iter=11, snapshot=img_query_data)
        else:
            erode_terrain3(points, neighbors, height, num_iter=11, snapshot=None)
        erode_end = time.perf_counter()
        print(f"Erosion runtime: {erode_end-erode_start:.5f}")

        # print(newheight)

        if args.png:
            export_list["height"] = height

# Climate Stuff
# =============================================
    if do_climate:

        # ====================
        # NOTE: Planetary average solar flux and balances

        # SBC = 5.670374419 * 10**-8  # Stefan-Boltzmann constant
        # 0.00000005670374419

        # Calculate the solar constant at the planet's orbit
        tsi = calculate_tsi(star_radius, star_temp, orbital_distance)

        # https://earthobservatory.nasa.gov/features/EnergyBalance/page1.php
        # TSI is at the top of the atmosphere. For Earth that's about 1370 watts per meter squared (W/m^2)
        # Because only half of the planet is lit, this immediately cuts that number in half.
        # Because a sphere isn't evenly lit (cosine falloff with latitude and longitude), this is cut in half again.
        # So, when averaged over the whole planet the incoming energy is 0.25 * TSI or ~ 342.5 W/m^2
        # But in addition to that some simply bounces away because of albedo. About 31% bounces away. 342 * (1 - 0.31) or 236.325 W/m^2 (note: I've seen sources say albedo is 29% to 31%)
        # 22% gets absorbed by the atmosphere by water vapor, dust, and ozone, and 47% gets absorbed by the surface. (so of the 69% that doesn't bounce, about 31.8% is absorbed in the atmosphere and 68.2% by the surface)

        earth_radius = 6378100.0  # NOTE: Manually keying this in for now because something is weird/wrong with visualizing the temperatures on a full-sized planet at the moment so I'm visualizing at radius 1.0 for the time being.

        planet_cross_section = np.pi * earth_radius**2
        planet_surface_area = 4 * np.pi * earth_radius**2  # Sunlight only falls on half of the surface area, however

        # ToDo: energy at the cross section, energy reduced by albedo, and maybe energy that bounces off the atmosphere? not sure if that's the same thing.  or maybe I meant absorbed by the atmosphere so it doesn't reach ground.
        # For basic temp assignment, I could start by calculated the airless body temperature and then just multiply by a fudge factor for greenhouse effect. (e.g. earth is like 34 C higher than it would be with no atmospheric blanket)
        # Basic energy balance in/out
        total_energy = tsi * planet_cross_section  # Total energy of the planet cross-section, in W
        energy_in = tsi * 0.25 * (1 - world_albedo)  # Energy in after factoring out albedo bounce and spherical geometry, in W/m^2
        atmosphere_watts = energy_in * 0.318  # After factoring out albedo bounce, this is how much the atmosphere absorbs in W/m^2
        surface_watts = energy_in * 0.682  # After factoring out albedo bounce, this is how much the surface absorbs in W/m^2

        # These values would probably be derived from the gas/aerosol composition of the atmosphere (I think the open source "Thrive" by Revolutionary Games does this; e.g. calculating light blockage via its wavelength in nanometers vs size of molecules)
        # Amount of incoming shortwave light frequencies that makes it down through the atmosphere
        shortwave_transmittance = 0.9
        # Amount of outgoing longwave IR frequencies that makes it out through the atmosphere
        longwave_transmittance = 0.16

        # There is a mismatch with what's written on the linked GHG1 page below
        # and what needs to be done. There is some poor/sloppy phrasing.
        # DON'T raise this to the power of 0.25. What looks like an earlier
        # version confirms this: http://homework.uoregon.edu/pub/class/es202/atm.html
        ground_flux = tsi * ((1 + shortwave_transmittance) / (1 + longwave_transmittance))  # Simple way to fudge atmospheric greenhouse blanket effect

        # Okay this equation here was a bit hard to figure out but its the
        # average surface temp for the whole planet without atmospherics.
        # The source is at the bottom of the following page:
        # http://homework.uoregon.edu/pub/class/es202/GRL/ghg1.html
        avg_surface_temp_unmodified = (tsi / (4*SBC))**0.25 * (1 - world_albedo)**0.25
        # Same but with atmospheric fudge plugged in
        avg_surface_temp_greenhouse = (ground_flux / (4*SBC))**0.25 * (1 - world_albedo)**0.25

        print("--   Star surface energy:", SBC * star_temp**4 * (4 * np.pi * star_radius**2))  # ToDo: Move this print into the calculate_tsi function.  (Also really need debug log levels to control verbosity of printing.)
        print("--                TSI is:", tsi)
        print("--Cross-sectional energy:", total_energy)
        print("--Ground flux is:        ", ground_flux)
        print("--Original surface temp: ", avg_surface_temp_unmodified)
        print("--Modified surface temp: ", avg_surface_temp_greenhouse)

        # ====================
        # NOTE: The original temperature assignment code

        print("Assigning starting temperatures...")
        temp_start = time.perf_counter()
        if snapshot_climate:
            surface_temps = assign_surface_temp(points, height, axial_tilt, surface_watts)  # ToDo: When climate sim actually has steps, implement snapshotting
        else:
            surface_temps = assign_surface_temp(points, height, axial_tilt, surface_watts)
        temp_end = time.perf_counter()
        print(f"Assign surface temps runtime: {temp_end-temp_start:.5f} sec")

        if args.png:
            export_list["surface_temp"] = surface_temps

        # ====================
        # NOTE: Some material equilibrium code

        edge_length = 1.0
        volume = edge_length**3  # X cubic meters  # NOTE: Consider whether it would be more relevant to do this with circles with an area of 1 meter squared, and/or cylinders of material.
        # volume = edge_length**2  # For water with a depth of only 1 meter
        # surface_area = edge_length**2  # Allow only 1 side of the cube to radiate energy
        surface_area = 6 * edge_length**2  # For a cube from edge length
        # surface_area = 6 * volume**(2/3)  # For a cube from volume
        # surface_area = (2*edge_length**2) + (4*edge_length)  # For water with a depth of only 1 meter
        emissivity = 1.0  # Not the actual value for water but we're pretending it's a perfect black-body
        water_mass = 1000 * volume  # water's density is 1000 kg/m^3
        water_heat_cap = 4184  # Water's heat capacity is 4184 Joules/kg/K
        water_albedo = 0.06

        sim_steps = 10  # Max number of steps. Will break early if equilibrium is reached.
        time_step = 3600  # W is J per second. This is the number of seconds for each step.
        if time_step == 1:
            time_units = "seconds"
        elif time_step == 60:
            time_units = "minutes"
        elif time_step == 3600:
            time_units = "hours"
        elif time_step == 86400:
            time_units = "days"
        elif time_step == 604800:
            time_units = "weeks"
        else:
            print("INVALID TIME STEP")
            sys.exit(0)

        # NOTE: There's an artifact if the starting temperature is above the eventual equilibrium temperature.
        # The temperature might actually spike upwards before then descending down to equilibrium,
        # or it might spike, then drop below equilibrium before rising to equilibrium.
        # This is due to multiplying the time_step in the new amount of TSI to add to the joules.
        # It causes a large sum of joules to be added initially before new_emit is calculated and
        # subtracted in the subsequent step(s) if the step is large.
        # Also, a large time_step can cause some of the variables to overflow because numbers get too big.
        old_temp = 200.0  # I'm scared to start at 0 kelvin so we'll start at 1
        old_joules = water_mass * water_heat_cap * old_temp# * time_step
        old_emit = SBC * old_temp**4 * surface_area * emissivity# * time_step
        print(" Water Temp is:", old_temp)
        print(" Starting joules:", old_joules)
        print(" Starting emit:", old_emit)

        # Also we're ignoring that water becomes ice below 273 K, and ice has a different heat capacity, albedo, etc.
        for i in range(sim_steps):  # Could do a while True
            old_joules = old_joules - old_emit + (tsi * (1-water_albedo) * edge_length**2 * time_step)
            # old_joules = old_joules - old_emit + (ground_flux * (1-water_albedo) * edge_length**2 * time_step)
            # print(" joules is:", old_joules)

            # From this page: https://www.e-education.psu.edu/earth103/node/1005
            # joules = mass * heat capacity * temperature
            # Therefore we can reverse engineer the temperature from joules / (mass * heat capacity)
            new_temp = old_joules / (water_mass * water_heat_cap)
            # print(" New temp:", new_temp)
            new_emit = SBC * new_temp**4 * surface_area * emissivity * time_step
            # print(" New emit:", new_emit)

            if round(old_temp, 8) == round(new_temp, 8):  # This is not the best method. i.e. large masses of material will change temp very slowly so this rounding will break early right at the start because the temp barely changed.
            # if old_temp == new_temp:
                print(f" Equilibrium ~= {new_temp}")
                break

            old_temp = new_temp
            old_emit = new_emit
            # if i % 24 == 0:
            print(f" Water Temp is: {new_temp:09.5f} at time step {i+1:.0f} {time_units}.")

            # Can we simply calculate the equillibrium directly?
            # water_mass * water_heat_cap * temperature = SBC * temperature**4 * surface_area * emissivity  # NOTE: This whole section is probably all kinds of wrong.
            # water_mass * water_heat_cap * temperature = SBC * (temperature*temperature*temperature*temperature) * surface_area * emissivity
            # water_mass * water_heat_cap = SBC * (temperature*temperature*temperature) * surface_area * emissivity
            # (temperature*temperature*temperature) = (SBC * surface_area * emissivity) / (water_mass * water_heat_cap)
            # temperature**3 = (SBC * surface_area * emissivity) / (water_mass * water_heat_cap)
            # temperature = ((SBC * surface_area * emissivity) / (water_mass * water_heat_cap))**(1/3)


        # ====================
        # NOTE: Solar flux at specific latitude, time, and day of year

        solstice = 173
        year_length = 365.25
        day_length = 24
        half_day = day_length / 2

        hour = 12
        day = 80
        latitude = 0
        longitude = 0

        axial_tilt = 23.44
        current_tilt = calculate_seasonal_tilt(axial_tilt, 90)

#        hour_angle = ( ((hour - half_day) * np.pi) / day_length ) + ((longitude * np.pi) / 180)
        # hour_angle = 90
        hour_angle = (360 / day_length) * (hour - half_day)  # In degrees

        declination = current_tilt * (np.pi/180) * np.cos( (2*np.pi * (day - solstice)) / year_length )

#        hour_angle = np.arccos(-np.tan(latitude) * np.tan(declination))# * np.pi/180

        # zenith = ( np.sin(latitude * (np.pi/180)) * np.sin(declination) + np.cos(latitude * (np.pi/180)) ) * ( np.cos(declination) * np.cos(hour_angle) )
        zenith = ( np.sin(latitude * (np.pi/180)) * np.sin(declination * (np.pi/180)) + np.cos(latitude * (np.pi/180)) ) * ( np.cos(declination * (np.pi/180)) * hour_angle )

        flux = tsi * np.cos(zenith * (np.pi/180))# * longwave_transmittance
        # flux = tsi * (1-world_albedo) * np.cos(zenith)# * longwave_transmittance

        test = tsi * np.cos(np.abs(latitude - current_tilt) * (np.pi/180)) * np.cos(hour_angle * (np.pi/180))  # Not quite right; when hour is 12 and lat = 0 it should be at full strength

        print(f"TSI is {tsi}")
        print(f"Hour angle is {hour_angle}")
        print(f"Declination angle is {declination}")
        print(f"Zenith angle is {zenith}")
        print(f"Flux at Lat: {latitude}, Lon: {longitude} at hour {hour} is: {flux}")
        print(f"Test is {test}")
        # ====================
        # NOTE: Attempt 2 at spherical solar flux.
        temps_2 = assign_surface_temp5(points, height, current_tilt, surface_watts)
        if args.png:
            export_list["surface_temp_2"] = temps_2


# Build KD Tree (and test query to show points on surface)
# =============================================
    if test_latlon:
        llpoint = latlon2xyz(-10, -18)
        if KDT is None:
            KDT = build_KDTree(points)

        print("Querying KD Tree for neighbors...")
        query_start = time.perf_counter()
        distances, neighbors = KDT.query(llpoint, 3)
        query_end = time.perf_counter()
        print(f"Query finished in {query_end - query_start :.5f} sec")

        print("neighbors:", neighbors)
        print("neighbor distances:", distances)
        print("neighbor xyz:", points[neighbors[0]], points[neighbors[1]], points[neighbors[2]])

# Save the world map to a texture file
# =============================================
    if args.png:
        print("Saving world map image...")

        for key, array in export_list.items():
            # print("before heights:", height)
            export_list[key] = rescale(array, 0, 255)
            # print("after heights: ", np.round(f_rescale.astype(int)))

        pixel_data = build_image_data(img_query_data, export_list)
        save_start = time.perf_counter()
        save_image(pixel_data, save_dir, world_name)
        save_end = time.perf_counter()
        print(f"Write to disk finished in  {save_end - save_start :.5f} sec")

        img_query_data = None
        pixel_data = None

# Visualize the final planet
# =============================================
    runtime_end = time.perf_counter()
    now = time.localtime()
    print(f"Computation finished at: {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}")
    print(f"Script runtime: {runtime_end - runtime_start:.3f} seconds")

    print("Preparing visualization...")
#    visualize(noise_vals, cells, height, search_point=llpoint, neighbors=neighbors)
#    visualize(points * (np.reshape(height, (len(points), 1)) + 1), cells, height)  # NOTE: Adding +1 to height means values only grow outward if range is 0-1. But for absolute meter ranges it merely throws the values off by 1.

    # NOTE: Adding +1 to height means values only grow outward if range is 0-1. But for absolute meter ranges it merely throws the values off by 1.
    # NOTE: Figure out the proper way to multiply points times the absolute heights to maintain the correct radius.
    # I think it's Verts * (1 + H/R)
    # points *= np.reshape(height + 1 + world_radius - 1, (len(points), 1))  # So, not this way

    points *= np.reshape(height/world_radius + 1, (len(points), 1))  # Rather, do it this way instead
    # visualize(points, cells, surface_temps, tilt=axial_tilt)
    visualize(points, cells, temps_2, tilt=axial_tilt)

    # ToDo: PyVista puts execution 'on hold' while it visualizes. After the user closes it execution resumes.
    # Consider asking the user right here if they want to save out the result as a png/mesh/point cloud.
    # (Only if those options weren't passed as arguments at the beginning, else do that arg automatically and don't ask)
    # (There's also a PyVista mode that runs in the background and can be repeatedly updated? Look into that.)

# Cleanup
# =============================================
    # temp_path = os.path.join(my_dir, "temp." + options["settings_format"])
    # if os.path.exists(temp_path):
    #     os.remove(temp_path)


@njit(cache=True, parallel=True, nogil=True)
def sample_noise(verts, perm, pgi, n_roughness=1, n_strength=0.2, radius=1):
    """Sample a simplex noise for given vertices"""
    elevations = np.ones(len(verts))

    rough_verts = verts * n_roughness

    for v in prange(len(rough_verts)):
        elevations[v] = osi.noise3d(rough_verts[v][0], rough_verts[v][1], rough_verts[v][2], perm, pgi)

    # print("Pre-elevations:")
    # print(elevations)
    return (elevations + 1) * 0.5 * n_strength * radius  # NOTE: I'm not sure if multiplying by the radius is the proper thing to do in my next implementation.
    # return elevations


def sample_octaves(verts, elevations, perm, pgi, n_octaves=1, n_init_roughness=1.5, n_init_strength=0.4, n_roughness=2.0, n_persistence=0.5, ocean_percent=0.5, world_radius=1.0):
    if elevations is None:
        elevations = np.zeros(len(verts))
    n_freq = n_init_roughness  # Frequency
    n_amp = n_init_strength  # Amplitude
    # In my separate-sampling experiment, rough/strength pairs of (1.6, 0.4) (5, 0.2) and (24, 0.02) were good for 3 octaves
    # The final 3 results were added and then multiplied by 0.4
    for i in range(n_octaves):
        print(f"\r  Octave {i+1}..", flush =True, end ='')
        time_start = time.perf_counter()
        elevations += sample_noise(verts, perm, pgi, n_freq / world_radius, n_amp / world_radius, world_radius)
        n_freq *= n_roughness
        n_amp *= n_persistence
        time_end = time.perf_counter()
        print(f" {time_end - time_start :.5f} sec")


    time_nmin = time.perf_counter()
    emin = np.amin(elevations)
    time_nmax = time.perf_counter()
    emax = np.amax(elevations)
    time_end = time.perf_counter()
    print(f"  Time to find numpy min: {time_nmax - time_nmin :.5f} sec")
    print(f"  Time to find numpy max: {time_end - time_nmax :.5f} sec")
    print("  min:", emin)
    print("  max:", emax)
    return elevations


def visualize(verts, tris, heights=None, search_point=None, neighbors=None, tilt=0.0):
    """Visualize the output."""
    # pyvista expects that faces have a leading number telling it how many
    # vertices a face has, e.g. [3, 0, 11, 5] where 3 means triangle.
    # https://docs.pyvista.org/examples/00-load/create-poly.html
    # So we fill an array with the number '3' and merge it with the cells
    # from meshzoo to get a proper array for pyvista.
    time_start = time.perf_counter()
    tri_size = np.full((len(tris), 1), 3)
    new_tris = np.hstack((tri_size, tris))
    time_end = time.perf_counter()
    print(f"Time to reshape triangle array: {time_end - time_start :.5f} sec")

    # pyvista mesh
    mesh = pv.PolyData(verts, new_tris)

    if search_point is not None and neighbors is not None:
        # Is it strictly necessary that these be np.arrays?
        neighbor_dots = pv.PolyData(np.array([verts[neighbors[0]], verts[neighbors[1]], verts[neighbors[2]]]))
        search_dot = pv.PolyData(np.array(search_point))
    x_axisline = pv.Line([-1.5,0,0],[1.5,0,0])
    y_axisline = pv.Line([0,-1.5,0],[0,1.5,0])
    z_axisline = pv.Line([0,0,-1.5],[0,0,1.5])

    # Axial tilt line
    # ax, ay, az = latlon2xyz(tilt, 45)
    ax, ay, az = latlon2xyz(tilt, 0)
    t_axisline = pv.Line([0,0,0], [ax * 1.5, ay * 1.5, az * 1.5])

    # Sun tilt line (line that is perpendicular to the incoming solar flux)
    # ax, ay, az = latlon2xyz(90-tilt, -135)
    ax, ay, az = latlon2xyz(90-tilt, 180)
    s_axisline = pv.Line([0,0,0], [ax * 1.5, ay * 1.5, az * 1.5])


    # Clip the heights again so we can show water separately from land gradient
    # by cleverly using the below_color for anything below what we clip here.
    minval = np.amin(heights)
    maxval = np.amax(heights)
    # heights = np.clip(heights, minval*1.001, maxval)

    # https://matplotlib.org/cmocean/
    # https://docs.pyvista.org/examples/02-plot/cmap.html
    # https://colorcet.holoviz.org/
    sargs = dict(below_label="Ocean")

    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=False, smooth_shading=True, color="white", below_color="blue", scalars=heights, cmap="thermal", culling = "back", scalar_bar_args=sargs)
    # pl.add_scalar_bar(below_label="Ocean")
    if search_point is not None and neighbors is not None:
        pl.add_mesh(neighbor_dots, point_size=15.0, color = "magenta")
        pl.add_mesh(search_dot, point_size=15.0, color = "purple")
    pl.add_mesh(x_axisline, line_width=5, color = "red")
    pl.add_mesh(y_axisline, line_width=5, color = "green")
    pl.add_mesh(z_axisline, line_width=5, color = "blue")
    pl.add_mesh(t_axisline, line_width=5, color = "magenta")
    pl.add_mesh(s_axisline, line_width=5, color = "yellow")
    pl.show_axes()
    print("Sending to PyVista.")
    pl.show()

# =============================================


if __name__ == '__main__':
    main()

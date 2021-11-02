"""A program to generate detailed maps for spherical worlds."""

import os
import sys
import argparse
import time
# import math
# import json
# import concurrent.futures
# import meshio as mi
# import meshzoo as mz
import numpy as np
from numba import njit, prange
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
# from scipy.spatial import KDTree
import opensimplex as osi
import cfg
from util import *
from terrain import sample_octaves, make_mask
from erosion import *
from climate import *
# pylint: disable=not-an-iterable
# pylint: disable=line-too-long

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
#    so they only need to precompute once.  Numpy has its own .npy and .npz formats, but of course they have to be read into RAM to actually use them. It could still be useful to juggle what is in RAM at any moment if things aren't needed.
#
# - Find out which takes up more bytes in RAM: color represented as RGB, HSV, or hex value, and if the 'winner' at a bit depth of 8 is still the winner at 16 or higher. (NOTE: I don't think hex can do > 8 bits)
#    I haven't looked into this, yet, but if each channel gets its own separate number that would be 3*N bytes times V number of verts.
#    For example a numpy float32 or int32 array is 4 bytes per item or 3*4*V, and a float64 or int64 array is 8 bytes per item or 3*8*V
#    1 byte:  bool_, int8, uint8
#    2 bytes: float16, int16, uint16
#    4 bytes: float32, int32, uint32
#    8 bytes: float64, int64, uint64
#    32 or 64 would be overkill for tracking 16 bits per channel, which is commonly stored as integer values, and covers a range of 0-65535, which matches the range of uint16; so 16-bits per channel would be 3 * 2 bytes * V
#    https://petebankhead.gitbooks.io/imagej-intro/content/chapters/bit_depths/bit_depths.html
#    There is also the option of floating point numbers which are used in HDR and such but that may be harder to work with and I'm not sure how much compatibility there is in common image editors.
#    Another possibility is to use a palette, like a gif, instead of full RGB saved per vertex. The vertex would store a number between 0-255, for example, and that index would look up the real RGB values in a palette with 256 colors.
#    A palette is something that will definitely be used for the simple map types like Biome, Climate, Metals, etc. because there are a limited number of biomes to choose from in a Whittaker diagram or Koppen climate classification.
#
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

# Style guide:
# - For functions that need mesh data like the vertices, the vertices should be the first parameter.
# - If the function takes an array that is indexed to the vertices, such as height or temperature, that should be the second parameter.
# - If it takes more than one array then those should be given in an order of importance that resembles the order in which they were created, such as height, then insolation, then temperature, then precipitation

class NixisPlanet:
    """Base class for Nixis planets."""  # This may or may not actually get used any time soon.
    def __init__(self):
        radius = None
        orbit_distance = None
        elevation = None
        surface_temp = None

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
    if (args.divisions % 2) == 0:
        divisions = args.divisions
    else:
        divisions = args.divisions + 1
        print(f"Even numbers only for divisions, please. Setting divisions to {divisions}.")
    if args.divisions > 2250:
        print("\n" + "WARNING. Setting divisions to a large value can use gigabytes of RAM.")
        print(f"         The 3D mesh will have {divisions * divisions * 10 + 2:,} vertices.")
        print(f"         The 3D mesh will have {divisions * divisions * 20:,} triangles.")
        print("         Simulation will also take proportionately longer." + "\n")
        confirm = input("Continue anyway? Y/N: ")
        if confirm.lower() not in ('y', 'yes'):
            sys.exit(0)

    if args.seed:
        world_seed = args.seed
    else:
        seed_rng = np.random.default_rng()
        world_seed = int(seed_rng.integers(low=0, high=999999))
        # world_seed = int(seed_rng.integers(low=-999999, high=999999))  # Negative numbers work fine, apparently
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

    options = load_settings("options.json")  # ToDo: If the user specifies args.png then we should grab the desired output maps from options.json. Right now the export_list dict is not being used at all.
    cfg.WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg.SAVE_DIR = os.path.join(cfg.WORK_DIR, options["save_folder"])
    cfg.SNAP_DIR = os.path.join(cfg.WORK_DIR, options["save_folder"], options["snapshot_folder"])
    img_width = options["img_width"]
    img_height = options["img_height"]
    export_list = options["export_list"]

    test_latlon = False

    do_erode = False
    do_climate = False

    snapshot_erosion = False
    snapshot_climate = False

    # ToDo: For both of the below we should possibly only try making the dir at the moment we save?
    # Test if the directory already exists. Maybe even attempt to see if we have write permission beforehand.
    # Actual exception types
    # if args.png or args.mesh or args.pointcloud or args.database or args.config:
    try:
        os.mkdir(cfg.SAVE_DIR)
    except:
        # print("Failed to create script output directory! Check folder permissions.")
        pass
    if snapshot_erosion or snapshot_climate:
        try:
            os.mkdir(cfg.SNAP_DIR)
        except:
            # print("Failed to create script output directory!")
            pass

    cfg.WORLD_CONFIG = {
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
        save_settings(cfg.WORLD_CONFIG, cfg.SAVE_DIR, world_name + '_config', fmt=options["settings_format"])

# Start the party
# =============================================
# =============================================
    runtime_start = time.perf_counter()
    now = time.localtime()
    print("==============================")
    print("World seed:", world_seed)
    print("\n" + f"Script started at: {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}" + "\n")

    # NOTE: For k=2500 the mesh takes 17 seconds to build. For k=5000 it takes 69 seconds. For k=7500 it takes 159 seconds.
    points, cells = create_mesh(divisions)  # points: float64, cells: int32
    if world_radius != 1:
        points *= world_radius

    # print("Points:")
    # print(points.dtype)
    # print("Cells:")
    # print(cells.dtype)

    # If I need to output several meshes for examination
    # for i in range(2, 26, 2):
    #     a, b = create_mesh(i)
    #     save_mesh(a, b, cfg.SAVE_DIR, f"{i:02}" + '_smooth')

    # Saving the mesh here just writes a plain sphere with no elevation offsets (for debug)
    # if args.mesh:
    #     save_mesh(points, cells, cfg.SAVE_DIR, world_name + '_smooth')

# Prepare KD Tree and stuff for image saving
# =============================================

    # ToDo: Test compact_nodes and balanced_tree args for build/query performance tradeoffs
    # NOTE: I've tested as high as k=7500 and the KD Tree took 264 seconds to build. Amazingly the query for the LL array only took half a second.
    # The Tree takes 114 seconds to build for k=5000, and 26 seconds for k=2500
    if args.png or snapshot_erosion or snapshot_climate:
        time_start = time.perf_counter()
        # Array of 3D coordinates on the sphere for each pixel that will be in the exported image.
        ll = make_ll_arr(img_width, img_height, world_radius)
        # NOTE: What if we did the reverse of this? Instead of converting 2D lat/lon of the pixels into 3D coordinates and then doing a KD Tree query in 3 dimensions,
        # what if we took the 3D coordinates of every mesh vertex and converted them into 2D coordinates and then did the nearest neighbor search in 2D?
        # Would a 2D search be faster? Would the edges of the 'image grid' be problematic because they wrap around to the other side? Would the poles have trouble finding 3 correct neighbors?
        ll_end = time.perf_counter()
        build_KDTree(points)
        kdt_end = time.perf_counter()
        # For each pixel's 3D coordinates, this is the 3 nearest vertices, and the distances to said vertices.
        # The name kinda sucks though. Maybe we could rename it img_pixel_neighbors.
        print("Building query...")
        cfg.IMG_QUERY_DATA = cfg.KDT.query(ll, k=3, workers=-1)
        query_end = time.perf_counter()

        ll = None
        if not test_latlon:
            cfg.KDT = None

        print(f"  LL built in       {ll_end - time_start :.5f} sec")
        print(f"  KD Tree built in  {kdt_end - ll_end :.5f} sec")
        print(f"  Query finished in {query_end - kdt_end :.5f} sec")


# Sample noise for initial terrain heights
# =============================================
    # Initialize the permutation arrays to be used in noise generation
    perm, pgi = osi.init(world_seed)  # Very fast. 0.02 secs or better

    # Ocean altitude is a *relative* percent of the range from min to max altitude.
    ocean_percent = 55.0
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

    # ToDo: Consider pulling arg values from cfg.py, options.json, seed json file, or as kwargs because this line is a long boi now.
    height = sample_octaves(points, None, perm, pgi, n_octaves, n_init_rough, n_init_strength, n_roughness, n_persistence, world_radius)

#    height = rescale(height, -0.05, 0.15)
    height = rescale(height, -4000, 8850)
#    height = rescale(height, -4000, 0, 0, mode='lower')
#    height = rescale(height, 0, 8850, 0, mode='upper')
#    height = rescale(height, -1, 1)

    # ToDo: Multiply the rescaled height times the world radius?
    minval = np.amin(height)
    maxval = np.amax(height)
    print("  Rescaled min:", minval)
    print("  Rescaled max:", maxval)

    ocean_level = find_percent_val(minval, maxval, ocean_percent)  # NOTE: Due to the way np.clip works (hack below), values less than 0% or greater than 100% have no effect.
    print("  Ocean Level:", ocean_level)

    ocean = make_mask(height, ocean_level)

    if args.png:
        export_list["ocean"] = [ocean, 'gray']

    # Bring ocean floors up with a power < 1
    height = power_rescale(height, mask=ocean, mode=1 , power=0.5)

    # Bring land down with a power > 1
    height = power_rescale(height, mask=ocean, mode=0 , power=2.0)


    # height *= 10
    # height += 6378100
    # height += 400000
    # height += world_radius - 1

#    height = np.clip(height, ocean_level, maxval)  # Hack for showing an ocean level in pyvista.  It might be time to try using a 2nd sphere mesh just for water, and use masks with the scalars or something so underwater areas don't bungle the height colormap.
    # Also this clipping should probably be one of the last steps before visualizing.
    # height = np.clip(height, minval, ocean_level)


    # height -= ocean_level  # Move the ocean level to 0; need to rescale the upper and lower halves again to regain proper min and max altitudes.

    # height = rescale(height, -0.05, 0.15, mid=0)
    # minval = np.amin(height)
    # maxval = np.amax(height)
    # print("  Rescaled min:", minval)
    # print("  Rescaled max:", maxval)

    # height = np.clip(height, 0, maxval)  # Can use 0 instead of ocean_level because we already rescaled right above here. Note that this has the effect of making the upper half steeper and the lower half flatter as the original bounds are restored.
    # NOTE: Maybe instead of subtracting the ocean level and doing another rescale, instead we only do that when multiplying the height times the vertices for the final viz.
    # The heights exist independently of the vertices until then. The main goal has been to get the sea level to match the sphere's radius.
    # The main issue is that the ocean level in the heights won't be 0, then. It will match the radius but will be higher than 0.
    # I think at the end of the day I need to implement a curve/power function for the rescaling like in that paper I can't find anymore, or RedBlob, to control the 'falloff' of the rescaling.
    # In RedBlob's example it's a simple power like for x in height: x = x**3, or x = (x * fudge_factor)**3, where powers between 0.1 to 1 will raise the middle elevations up towards mountain peaks and powers > 1 (like 3) push mids down into valley floors.
    # It should be noted that the simple power function only works properly when values are between 0 and 1, otherwise values > 1 will shoot off into space or be smashed down.
    # So getting the terrain curves right is probably going to require multiple wacky rescale operations.
    # Continental shelves are probably going to need a power < 1 to raise them close to the land level, but ocean floor after the shelf will need a power > 1 to lower and flatten it with falloff from the shelf. Land needs a power > 1 to slope upward from coasts.

    # height *= world_radius  # Keeps the scale relative. NOTE: But don't multiply here, because that messes up the display of the scalar bar in pyvista. Instead, do this multiplication down inside the visualize function.

    if args.png and not do_erode:
        #potato = rescale(height, -4000, 8850) + 32768
        # export_list["height"] = [potato.astype('uint16'), 'gray']
        # export_list["height"] = [(rescale(height, -4000, 8850) + 32768).astype('uint16'), 'gray']
        export_list["height_absolute"] = [ (rescale(height, -4000, 8850) + (32768 - find_percent_val(-4000, 8850, ocean_percent))).astype('uint16') , 'gray']
        print("THE THING MIN:", np.min(export_list["height_absolute"][0]))
        print("THE THING MAX:", np.max(export_list["height_absolute"][0]))
        export_list["height_relative"] = [ (rescale(height, 0, 65535)).astype('uint16') , 'gray']

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
        erode_terrain3(points, neighbors, height, num_iter=11, snapshot=snapshot_erosion)  # ToDo: Placing the neighbors before the height probably violates my style guide.
        erode_end = time.perf_counter()
        print(f"Erosion runtime: {erode_end-erode_start:.5f}")

        # print(newheight)

        if args.png:
            export_list["height"] = [height, 'gray']

# Climate Stuff
# =============================================
    axial_tilt = 23.44
    current_tilt = calculate_seasonal_tilt(axial_tilt, 36)

    if do_climate:

        # Calculate the solar constant at the planet's orbit
        tsi = calculate_tsi(star_radius, star_temp, orbital_distance)

        # Calculate equilibrium temperature for the planet
        # calc_planet_equilibrium(tsi, world_albedo)

        # Calculate equilibrium temperature for a block of water
        # calc_water_equilibrium(tsi)

        # ====================
        # NOTE: The original temperature assignment code

        print("Assigning starting temperatures...")
        temp_start = time.perf_counter()
        surface_temps = assign_surface_temp(points, height, world_radius, current_tilt)
        temp_end = time.perf_counter()
        print(f"Assign surface temps runtime: {temp_end-temp_start:.5f} sec")

        if args.png:
            export_list["surface_temp"] = [surface_temps, 'gray']

        # ====================
        # NOTE: Attempt 1 at spherical solar flux (hour angle stuff); big fail.
        # calc_hour_angle_insolation(tsi)

        # NOTE: Instant insolation
        do_instant = True
        if do_instant:  # ToDo: Work out the relationship between time of day and rotation at any longitude so we can specify a time at a location and automatically rotate by the correct amount.
            print("Calculating solar insolation snapshot in time...")
            temp_start = time.perf_counter()
            insol_2 = calc_instant_insolation(points, height, world_radius, 0, current_tilt)
            temp_end = time.perf_counter()
            print(f"Runtime: {temp_end-temp_start:.5f} sec")
            if args.png:
                export_list["instant_insol"] = [insol_2, 'gray']

        # NOTE: Version '2.5'
        print("Calculating daily solar insolation (slice method)...")
        temp_start = time.perf_counter()
        daily_insolation = calc_daily_insolation(points, height, world_radius, current_tilt)
        temp_end = time.perf_counter()
        print(f"Runtime: {temp_end-temp_start:.5f} sec")

        # daily_insolation *= (tsi/360)  # Compared to the NASA data for 1980 this is off by like 20 Watts or so, but the cosine angle is basically perfect.
        if args.png:
            export_list["daily_insolation"] = [daily_insolation, 'gray']

        do_annual = False
        if do_annual:
            print("Calculating annual solar insolation (slice method)...")
            temp_start = time.perf_counter()
            annual_insolation = calc_yearly_insolation(points, height, world_radius, axial_tilt, snapshot_climate)
            temp_end = time.perf_counter()
            print(f"Runtime: {temp_end-temp_start:.5f} sec")

            if args.png:
                export_list["annual_insolation"] = [annual_insolation, 'gray']

        # calc_equilibrium_temp()


# Build KD Tree (and test query to show points on surface)
# =============================================
    if test_latlon:
        llpoint = latlon2xyz(20, 15, world_radius)
        if cfg.KDT is None:
            build_KDTree(points)

        # approx_size(getsize(cfg.KDT), "KD Tree", "BINARY")

        print("Querying KD Tree for neighbors...")
        query_start = time.perf_counter()
        lldistances, llneighbors = cfg.KDT.query(llpoint, 3)
        query_end = time.perf_counter()
        print(f"Query finished in {query_end - query_start :.5f} sec")

        print("neighbors:", llneighbors)
        print("neighbor distances:", lldistances)
        print("neighbor xyz:", points[llneighbors[0]], points[llneighbors[1]], points[llneighbors[2]])

# Save the world map to a texture file
# =============================================
    if args.png:
        print("Saving world map image(s)...")

        # ToDo: 16-bit images (rescale 0-65535 with 32768 as sea level--if heights are absolute already, just add 32768--and save support in the save_image util function)
        # (For RGB this may require abandoning the pillow library but for grayscale pillow should still work)
        # Idea: We could get half-meter precision if we have an export mode where we export bathymetry separately from land elevations;
        # multiply the input array * 2 before rounding to integers, I think. Elevation starts at 0 and goes up, bathymetry starts at 65535 and goes down.

        # I'm kind of thinking that maybe the CFG module should have its own dict that records what each array's mode is.. like, get the key from export_list, then like cfg.img_modes[key] gets the dtype?
        # But how do we determine the appropriate range to rescale? Or rather are there times when rescaling to the dtype is not wanted? If the dtype is uint16 then presumably it's a heightmap, but what if I want temperature in 16-bit?
        # Temperature ALSO has positive and negative values, though, so maybe insolation or precipitation would be a better example, because those can't have negative values.
        # Maybe a simple dummy class for each array like how in the Context Select add-on for Blender the ObjectMode class works. So for Nixis it'd be like cfg.height.mode (RGB, gray, etc), cfg.height.dtype, cfg.height.minmax or .rescale or something?
        # Would python allow it to be crammed into one class?  Like cfg.img.height.dtype (wow that's unfortunately long)
        # It might just be best to restructure the export_list and the functions that handle it (aka build_image_data) to be a dict with a list inside like {"height":[height_array, mode, dtype]}
        # for key, container in export_list.items():  container[0] is always the array, container[1] is always the mode, container[2] is always the dtype, and so on.
        # Or instead of calling the parameter "mode", call it "channels" because that's basically what it amounts to; the number of channels the image will have. 1 for grayscale, 3 or 4 for RGB or RGBA, and the dtype controls bit depth.
        # OR, the shit could just be properly rescaled and stuff at the end of the relevant section in nixis.py
        # i.e. at the part where I go if args.png: export_list["height"] = height  instead becomes, like, export_list["height"] = height.astype('uint16) + 32768  or something along those lines.  Then we don't have to guess later what to do.
        # I THINK that would preserve the original height array to be used in erosion and pyvista, and would create a copy in the export_list.  This can be tested by checking the python id for the objects, or just printing the numpy min/max for each.

        pixel_data = build_image_data(export_list)
        export_list = None

        save_start = time.perf_counter()
        save_image(pixel_data, cfg.SAVE_DIR, world_name)
        save_end = time.perf_counter()
        print(f"Write to disk finished in  {save_end - save_start :.5f} sec")

        cfg.IMG_QUERY_DATA = None
        pixel_data = None

# Visualize the final planet
# =============================================
    runtime_end = time.perf_counter()
    now = time.localtime()
    print("\n" + f"Computation finished at: {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}")
    print(f"Script runtime: {runtime_end - runtime_start:.3f} seconds" + "\n")

    print("Preparing visualization...")
    # The array that we want to use for the scalar bar.
    scalars = {"s-mode":"height", "scalars":height}
    # Possible s-modes to control the gradient used by the scalar bar:
        # height (solid color for ocean, colors for land)
        # bathy (colors for ocean height, solid color for land)
        # ocean (some other ocean metric colors, solid color for land)
        # topo (gradient has a hard break at the shore from ocean to land)
        # surface temperature or surface insolation (which is uniform for the full surface from min to max without weird breaks at the land/ocean border)
    # visualize(points, cells, height, scalars, search_point=llpoint, neighbors=llneighbors, radius=world_radius, tilt=current_tilt)
#    visualize(points * (np.reshape(height, (len(points), 1)) + 1), cells, height, scalars, tilt=current_tilt)  # NOTE: Adding +1 to height means values only grow outward if range is 0-1. But for absolute meter ranges it merely throws the values off by 1.

    # NOTE: Adding +1 to height means values only grow outward if range is 0-1. But for absolute meter ranges it merely throws the values off by 1.
    # NOTE: Figure out the proper way to multiply points times the absolute heights to maintain the correct radius.
    # I think it's Verts * (1 + H/R)
    # points *= np.reshape(height + 1 + world_radius - 1, (len(points), 1))  # So, not this way

    # print("Before points")
    # print(points)

    # Reshaping the heights to match the shape of the vertices array so we can multiply the verticies * the heights.
    # points *= np.reshape(height/world_radius + 1, (len(points), 1))  # Rather, do it this way instead. NOTE: This should really be done inside the visualize function.
    # https://gamedev.net/forums/topic/316602-increase-magnitude-of-a-vector-by-a-value/3029750/
    points *= np.reshape((height-ocean_level)/world_radius + 1, (len(points), 1))  # Rather, do it this way instead. NOTE: This should really be done inside the visualize function.

    # print("After points")
    # print(points)

    if test_latlon:
        visualize(points, cells, height, scalars, zero_level=ocean_level, search_point=llpoint, neighbors=llneighbors, radius=world_radius, tilt=current_tilt)
    else:
        visualize(points, cells, height, scalars,  zero_level=ocean_level, radius=world_radius, tilt=current_tilt)

    # ToDo: PyVista puts execution 'on hold' while it visualizes. After the user closes it execution resumes.
    # Consider asking the user right here if they want to save out the result as a png/mesh/point cloud.
    # (Only if those options weren't passed as arguments at the beginning, else do that arg automatically and don't ask)
    # (There's also a PyVista mode that runs in the background and can be repeatedly updated? Look into that.)

# Cleanup
# =============================================
    # temp_path = os.path.join(cfg.SAVE_DIR, world_name + '_config.' + options["settings_format"])
    # if os.path.exists(temp_path):
    #     os.remove(temp_path)


def visualize(verts, tris, heights=None, scalars=None, zero_level=0.0, search_point=None, neighbors=None, radius=1.0, tilt=0.0):
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

    # Create pyvista mesh from our icosphere
    mesh = pv.PolyData(verts, new_tris)
    # Separate mesh for ocean water
    ocean_shell = pv.ParametricEllipsoid(radius, radius, radius, u_res=300, v_res=300)

    if search_point is not None and neighbors is not None:
        # Is it strictly necessary that these be np.arrays?
        neighbor_dots = pv.PolyData(np.array([verts[v] for v in neighbors]))
        search_dot = pv.PolyData(np.array(search_point))
    x_axisline = pv.Line([-1.5*radius,0,0],[1.5*radius,0,0])
    y_axisline = pv.Line([0,-1.5*radius,0],[0,1.5*radius,0])
    z_axisline = pv.Line([0,0,-1.5*radius],[0,0,1.5*radius])

    # Axial tilt line
    # ax, ay, az = latlon2xyz(tilt, 45, radius)
    ax, ay, az = latlon2xyz(tilt, 0, radius)
    t_axisline = pv.Line([0,0,0], [ax * 1.5, ay * 1.5, az * 1.5])

    # Sun tilt line (line that is perpendicular to the incoming solar flux)
    # ax, ay, az = latlon2xyz(90-tilt, -135, radius)
    ax, ay, az = latlon2xyz(90-tilt, 180, radius)
    s_axisline = pv.Line([0,0,0], [ax * 1.5, ay * 1.5, az * 1.5])

    # Clip the heights again so we can show water separately from land gradient
    # by cleverly using the below_color for anything below what we clip here.
    minval = np.amin(heights)
    maxval = np.amax(heights)
    # heights = np.clip(heights, minval*1.001, maxval)

    # ===============
    # Define the colors we want to use
    blue = np.array([12/256, 238/256, 246/256, 1])
    black = np.array([11/256, 11/256, 11/256, 1])
    grey = np.array([189/256, 189/256, 189/256, 1])
    yellow = np.array([255/256, 247/256, 0/256, 1])
    red = np.array([1, 0, 0, 1])

    # Derive percentage of transition from ocean to land from zero level
    cmap_zl1 = ( (zero_level - minval) / (maxval - minval) ) - 0.001
    cmap_zl0 = cmap_zl1 - 0.001
    print("cmap transition lower:", cmap_zl0)
    print("cmap transition upper:", cmap_zl1)

    custom_cmap = LinearSegmentedColormap.from_list('ocean_and_topo', [(0, [0.1,0.2,0.6]), (cmap_zl0, [0.8,0.8,0.65]), (cmap_zl1, [0.3,0.4,0.0]), (1, [1,1,1])])
    # ===============

    # https://matplotlib.org/cmocean/
    # https://docs.pyvista.org/examples/02-plot/cmap.html
    # https://colorcet.holoviz.org/
    # sargs = dict(below_label="Ocean", n_labels=0, label_font_size=15)
    sargs = dict(n_labels=0, label_font_size=12, position_y=0.07)
    # anno = {minval:f"{minval:.2}", zero_level:"0.00", maxval:f"{maxval:.2}"}
    anno = {minval:f"{minval:.2}", find_percent_val(minval, maxval, cmap_zl0*100):"0.00", maxval:f"{maxval:.2}"}

    # ToDo: Add title to the scalar bar sargs and dynamically change it based on what is being visualized (e.g. Elevation, Surface Temperature, etc.)
    # title="whatever" (remove the quotes and make 'whatever' into a variable, like the s-mode or whatever. like title=scalars["s-mode"])
    # "Current"? ".items()"? https://stackoverflow.com/questions/3545331/how-can-i-get-dictionary-key-as-variable-directly-in-python-not-by-searching-fr
    # https://stackoverflow.com/questions/16819222/how-to-return-dictionary-keys-as-a-list-in-python

    pl = pv.Plotter()
    # pl.add_mesh(mesh, show_edges=False, smooth_shading=True, color="white", below_color="blue", culling="back", scalars=scalars["scalars"], cmap=custom_cmap, scalar_bar_args=sargs, annotations=anno)
    pl.add_mesh(mesh, show_edges=False, smooth_shading=True, color="white", culling="back", scalars=scalars["scalars"], cmap=custom_cmap, scalar_bar_args=sargs, annotations=anno)
    pl.add_mesh(ocean_shell, show_edges=False, smooth_shading=True, color="blue", opacity=0.15)

    if search_point is not None and neighbors is not None:
        pl.add_mesh(neighbor_dots, point_size=15.0, color = "magenta")
        pl.add_mesh(search_dot, point_size=15.0, color = "purple")
    pl.add_mesh(x_axisline, line_width=5, color = "red")
    pl.add_mesh(y_axisline, line_width=5, color = "green")
    pl.add_mesh(z_axisline, line_width=5, color = "blue")
    pl.add_mesh(t_axisline, line_width=5, color = "magenta")
    pl.add_mesh(s_axisline, line_width=5, color = "yellow")
    pl.show_axes()
    pl.enable_terrain_style(mouse_wheel_zooms=True)  # Use turntable style navigation
    print("Sending to PyVista.")
    pl.show()

# =============================================


if __name__ == '__main__':
    main()

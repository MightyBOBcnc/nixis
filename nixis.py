"""A program to generate detailed maps for spherical worlds.
Use -h or --help for arguments."""

import os
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
# from numba import njit, prange
# import pyvista as pv
import opensimplex as osi
import cfg
from util import *
from gui import visualize
from terrain import sample_octaves, make_bool_elevation_mask
from erosion import *
from climate import *
# pylint: disable=not-an-iterable
# pylint: disable=line-too-long

# ToDo List:
# - In the future when we have climate simulation, run a test to see if it's faster to lookup xyz2latlon as-needed or to simply store latlon for each vertex in a big master array.
#    NOTE: RAM considerations. Instead of storing it in RAM--and this is something applicable to all arrays--to limit RAM footprint it might be better to store most arrays on disk (such as in sqlite) and read values
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
# - In addition to speed profiling, also do some memory profiling because this could end up needing many GB of RAM. (the verts + tris alone need 12.6 GB of RAM--EACH--when k=7500)
# - Note: Voroni tesselation library: https://github.com/gdmcbain/voropy
# - Idea: Meshzoo is already really fast; I wonder if it can be @njitted (Answer: Not easily)
# - One thing to test is that it may be faster to use predefined vertex, triangle, and adjacency arrays that we read from disk instead of building the mesh with meshzoo and building the adjacency every single time.
#    This would mean there is less flexibility in choosing what level of subdivision to use when creating worlds but it might save on computation time and possibly on RAM since we don't have to make various copies
#    of arrays to build adjacency and such.  Maybe not so useful while I'm building the program but it could be useful once the program is 'production ready'.
#    I could optionally ship the program with precomputed arrays for download or include a precompute_arrays() function so the user can do it themselves (depending on how large the arrays are.. they'd probably be huge on disk)
#    so they only need to precompute once.  Numpy has its own .npy and .npz formats, but of course they have to be read into RAM to actually use them. It could still be useful to juggle what is in RAM at any moment if things aren't needed.
#    (Although numpy needs contiguous blocks of memory so things could potentially get messy if we're juggling RAM and setting things to None.)
#
# - Find out which takes up more bytes in RAM: color represented as RGB, HSV, or hex value, and if the 'winner' at a bit depth of 8 is still the winner at 16 or higher. (NOTE: I don't think hex can do > 8 bits)
#    I haven't looked into this, yet, but if each channel gets its own separate number that would be 3 channels, times N bytes, times V number of verts, or 3*N*V.
#    For example a numpy float32 or int32 array is 4 bytes per item or 3*4*V, and a float64 or int64 array is 8 bytes per item or 3*8*V
#    1 byte:  bool_, int8, uint8
#    2 bytes: float16, int16, uint16
#    4 bytes: float32, int32, uint32
#    8 bytes: float64, int64, uint64
#    32 or 64 would be overkill for tracking 16 bits per channel, which is commonly stored as integer values, and covers a range of 0-65535, which matches the range of uint16; so 16-bits per channel would be 3 * 2 bytes * V
#    https://petebankhead.gitbooks.io/imagej-intro/content/chapters/bit_depths/bit_depths.html
#    There is also the option of floating point numbers which are used in HDR and such but that may be harder to work with and I'm not sure how much compatibility there is in common image editors (or in the Pillow library).
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
# -- If it also needs the triangle data, then the triangle array should come before the others.
# - If it takes more than one array then those should be given in an order of importance that resembles the order in which they were created, such as height, then insolation, then temperature, then precipitation
# - Functions that create new data should be prefixed with "make_" as part of their snake_case name--e.g. make_rgb_array, make_mesh, etc.--as this is shorter than "build_", "create_", etc..

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
        mesh with ~2 million triangles. 2500 makes a 125mil triangle mesh. \
        Default is 320.")
    parser.add_argument("-s", "--seed", type=int,
        help="A number used to seed the RNG. If not specified then a random \
        number will be used.")
    parser.add_argument("-r", "--radius", type=float,
        help="Planet radius in meters. Default is Earth's radius.")
    parser.add_argument("-t", "--tilt", type=float, default=0.0,
        help="Axial tilt of the planet, in degrees.")
    # parser.add_argument("-lc", "--loadconfig", type=str,
    #     help="Load the generation settings from a text file. NOTE: Any other \
    #     provided arguments will override individual settings from the file.")
    parser.add_argument("--save_mesh", action="store_true",
        help="Export the world as a 3D mesh file.")
    # parser.add_argument("--save_pointcloud", action="store_true",
    #     help="Export the world as a 3D point cloud file.")
    # parser.add_argument("--save_database", action="store_true",
    #     help="Export the world as a sqlite file.")
    parser.add_argument("--save_img", action="store_true",
        help="Save a png map of the world.")
    parser.add_argument("--save_config", action="store_true",
        help="Save the generation settings for a given world as a text file.")
    parser.add_argument("--novis", action="store_true",
        help="Run Nixis without visualizing the final result in PyVista. \
        You may want to use this option for very large meshes or if you are \
        automating the export of maps without wanting to use the 3D viewer.")
    # ToDo: Arg for user-defined output path\folder.

    # Dict to store any marker points we'd like to render
    surface_points = defaultdict(list)
    test_latlon = False

    do_erode = True
    do_climate = False

    snapshot_erosion = True
    snapshot_climate = False

    args = parser.parse_args()

    if args.divisions <= 0:
        print("\n" + "Divisions must be an even, positive integer." + "\n")
        sys.exit(0)
    if (args.divisions % 2) == 0:
        divisions = args.divisions
    else:
        divisions = args.divisions + 1
        print(f"Even numbers only for divisions, please. Setting divisions to {divisions}.")
    # ToDo: Let's try to estimate RAM usage based on what arguments have been passed (divisions, whether to export images, and climate/erosion flags) so we can add that to the warning.
    # https://www.thepythoncode.com/article/get-hardware-system-information-python
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

    if args.radius:  # NOTE: Could do like "world_radius = args.radius if args.radius else 1.0" to save a few lines of code
        world_radius = abs(args.radius)
    else:  # NOTE: Technically speaking I should do the same as I did for divisions and simply set a default in argparse and not have this if/else statement at all.
        world_radius = 6378100.0  # Actual earth radius in meters
        # world_radius = 1.0  # In meters

    if -90.0 <= args.tilt <= 90.0:  # ToDo: Make sure that the temperature and daily rotation calculations aren't fragile and then expand this to allow larger tilts.
        axial_tilt = args.tilt      # e.g. Uranus and Venus have tilts > 90 (97 and 177, respectively) and several planets have retrograde rotation.
    else:
        print("\n" + "Axial tilt must be in the range -90 to +90." + "\n")
        sys.exit(0)

    world_albedo = 0.31  # Approximate albedo of Earth
    orbital_distance = 149597870.7  # 1 AU in km
    star_radius = 696340  # Sol's radius in km
    star_temp = 5778  # Sol's surface temperature in Kelvin

    options = load_settings("options.json")  # ToDo: If the user specifies args.save_img then we should grab the desired output maps from options.json. Right now the export_list dict is not being used at all.
    cfg.WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg.SAVE_DIR = os.path.join(cfg.WORK_DIR, options["save_folder"])
    cfg.SNAP_DIR = os.path.join(cfg.WORK_DIR, options["save_folder"], options["snapshot_folder"])
    img_width = options["img_width"]
    img_height = options["img_height"]
    export_maps = options["export_list"]  # Maps to export. (not used at the moment)

    # Dict to store world data for image export
    export_list = {}

    # ToDo: For both of the below we should possibly only try making the dir at the moment we save?
    # Test if the directory already exists. Maybe even attempt to see if we have write permission beforehand.
    # ToDo: Actual exception types
    # Also eventually these should perhaps be arguments for argparse, like --snap_e, --snap_c, or --snapshot_erosion, --snapshot_climate
    # if args.save_img or args.save_mesh or args.save_pointcloud or args.save_database or args.save_config:
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
    if args.save_config:
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
    # ToDo: Consider searching for a library or algorithm that is faster and possibly that can generate only the vertices without the triangles (unsure if that's feasible; at the very least you do need to edge pairs for subdividing)
    # in order to save RAM and avoid unnecessary computation; also one that can be @njit decorated or that has a C/C++ binary with a python interface.
    # Or take a look at forking and stripping down the icosa_sphere function from meshzoo to only the parts that are needed and try @njit decoration again. (parts of it have optional arguments that are not needed)
    # (NOTE: meshzoo is GPL v3; so if I do a stripped fork it'd be wise to use a separate repo and have it be a dependency)
    points, cells = create_mesh(divisions)  # points: float64, cells: int32
    if world_radius != 1:
        points *= world_radius

    # Cells are not being used for anything if there's no erosion and no visualization
    if args.novis and not do_erode:
        cells = None

    # If I need to output several meshes for examination
    # for i in range(2, 26, 2):
    #     a, b = create_mesh(i)
    #     save_mesh(a, b, cfg.SAVE_DIR, f"{i:02}" + '_smooth')

    # Saving the mesh here just writes a plain sphere with no elevation offsets (for debug)
    # if args.save_mesh:
    #     save_mesh(points, cells, cfg.SAVE_DIR, world_name + '_smooth')

    # Show the locations of the 12 verts with connectivity 5 for dev/debug purposes
    surface_points["cyan"].extend(points[:12])

# Prepare KD Tree and stuff for image saving
# =============================================

    # ToDo: Test compact_nodes and balanced_tree args for build/query performance tradeoffs
    # NOTE: I've tested as high as k=7500 and the KD Tree took 264 seconds to build. Amazingly the query for the LL array only took half a second.
    # The Tree takes 114 seconds to build for k=5000, and 26 seconds for k=2500
    if args.save_img or snapshot_erosion or snapshot_climate:
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
        #Performance (k=320):
        # 2k * 1k image:   0.15 seconds
        # 4k * 2k image:   0.65 seconds
        # 8k * 4k image:   2.4 seconds
        # 16k * 8k image:  14 seconds
        # 32k * 16k image: 20 seconds
        query_end = time.perf_counter()

        # Visualize the 2D pixel locations as coordinates on the 3D planet
        # for row in ll:
        #     surface_points["red"].extend(row)

        ll = None
        if not test_latlon:
            cfg.KDT = None

        print(f"  LL built in       {ll_end - time_start :.5f} sec")
        print(f"  KD Tree built in  {kdt_end - ll_end :.5f} sec")
        print(f"  Query finished in {query_end - kdt_end :.5f} sec")


# Sample noise for initial terrain heights
# =============================================
    # Initialize the permutation arrays to be used in noise generation
    # ToDo: Test the updated opensimplex library and see if it has good performance so we wouldn't have to use our custom fork.
    # https://github.com/lmas/opensimplex/issues/4#issuecomment-934671121
    perm, pgi = osi.init(world_seed)  # Very fast. 0.02 secs or better

    # min_alt = -0.05
    # max_alt = 0.15
    min_alt = -4000
    max_alt = 8850
    # Ocean percent is a *relative* percent of the range from min to max altitude.
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

    height = rescale(height, min_alt, max_alt)
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

    ocean = make_bool_elevation_mask(height, ocean_level)

    if args.save_img:
        export_list["ocean"] = [ocean, 'gray']

    # Bring ocean floors up with a power < 1
    height = power_rescale(height, mask=ocean, mode=1 , power=0.5)

    # Bring land down with a power > 1
    height = power_rescale(height, mask=ocean, mode=0 , power=2.0)

#    height = np.clip(height, ocean_level, maxval)  # Hack for showing an ocean level in pyvista.  It might be time to try using a 2nd sphere mesh just for water, and use masks with the scalars or something so underwater areas don't bungle the height colormap.
    # Also this clipping should probably be one of the last steps before visualizing.
    # height = np.clip(height, minval, ocean_level)


    height -= ocean_level  # Move the ocean level to 0; need to rescale the upper and lower halves again to regain proper min and max altitudes.

    height = rescale(height, min_alt, max_alt, mid=0)
    minval = np.amin(height)
    maxval = np.amax(height)
    print("  Rescaled min:", minval)
    print("  Rescaled max:", maxval)

    # height = np.clip(height, 0, maxval)  # Can use 0 instead of ocean_level because we already rescaled right above here. Note that this has the effect of making the upper half steeper and the lower half flatter as the original bounds are restored.
    # NOTE: Maybe instead of subtracting the ocean level and doing another rescale, instead we only do that when multiplying the height times the vertices for the final viz.
    # The heights exist independently of the vertices until then. The main goal has been to get the sea level to match the sphere's radius.
    # The main issue is that the ocean level in the heights won't be 0, then. It will match the radius but will be higher than 0.
    # I think at the end of the day I need to implement a curve/power function for the rescaling like in that paper I can't find anymore, or RedBlob, to control the 'falloff' of the rescaling.
    # In RedBlob's example it's a simple power like for x in height: x = x**3, or x = (x * fudge_factor)**3, where powers between 0.1 to 1 will raise the middle elevations up towards mountain peaks and powers > 1 (like 3) push mids down into valley floors.
    # It should be noted that the simple power function only works properly when values are between 0 and 1, otherwise values > 1 will shoot off into space or be smashed down.
    # So getting the terrain curves right is probably going to require multiple wacky rescale operations.
    # Continental shelves are probably going to need a power < 1 to raise them close to the land level, but ocean floor after the shelf will need a power > 1 to lower and flatten it with falloff from the shelf. Land needs a power > 1 to slope upward from coasts.

#    height *= world_radius  # Keeps the scale relative. NOTE: But don't multiply here, because that messes up the display of the scalar bar in pyvista. Instead, do this multiplication down inside the visualize function.

    if args.save_img and not do_erode:
        #potato = rescale(height, -4000, 8850) + 32768
        # export_list["height"] = [potato.astype('uint16'), 'gray']
        # export_list["height"] = [(rescale(height, -4000, 8850) + 32768).astype('uint16'), 'gray']
        # export_list["height_absolute"] = [ (rescale(height, -4000, 8850) + (32768 - find_percent_val(-4000, 8850, ocean_percent))).astype('uint16') , 'gray']
        export_list["height_absolute"] = [ (height + 32768).astype('uint16'), 'gray']
        print("Absolute Min:", np.min(export_list["height_absolute"][0]))
        print("Absolute Max:", np.max(export_list["height_absolute"][0]))
        export_list["height_relative"] = [ (rescale(height, 0, 65535)).astype('uint16') , 'gray']

# Erode!
# ToDo: Which came first, the erosion or the climate? Maybe do climate first, as the initial precipitation determines erosion rates.
#   Temperature also determines evaporation rate which affects how much erosion we can do.
#   And if we're being fancy with Thermal Erosion instead of approximating then we need surface temps first.
#   However then the altitude-based temperature will be wrong, so, climate has to run again after erosion. (Really they probably need to alternate)
# =============================================
    if do_erode:
        neighbors = build_adjacency(cells)
#        sort_adjacency(neighbors)

        # print(height)
        # print(type(height[0][0]))
        # ToDo: Should probably have an ocean clip mask that I can pass through for
        # verts that should be ignored during erosion, although if land borders
        # ocean then it should be able to add to the ocean floor to some degree
        # until uplifts above the ocean level, at which point it is removed from
        # the mask and becomes part of the land points.

        erode_start = time.perf_counter()
        average_terrain2(cells, height, num_iter=1)
        erode_terrain5(points, neighbors, height, num_iter=200, snapshot=snapshot_erosion)  # ToDo: Placing the neighbors before the height probably violates my style guide.
        # average_terrain2(cells, height, num_iter=3)
        # average_terrain_weighted(points, cells, height, num_iter=30)  # No real benefit to this over a simple average as far as I can tell.
        erode_end = time.perf_counter()
        print(f"Erosion runtime: {erode_end-erode_start:.5f}")

        # print(newheight)

        # ToDo: The ocean mask needs to be updated during/after erosion.
        #  At the moment it isn't updated after initial creation from the
        #  simplex noise height.

        if args.save_img:
            # export_list["height"] = [height, 'gray']
            export_list["height_absolute"] = [ (height + 32768).astype('uint16'), 'gray']
            print("Absolute Min:", np.min(export_list["height_absolute"][0]))
            print("Absolute Max:", np.max(export_list["height_absolute"][0]))
            export_list["height_relative"] = [ (rescale(height, 0, 65535)).astype('uint16') , 'gray']

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

        if args.save_img:
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
            if args.save_img:
                export_list["instant_insol"] = [insol_2, 'gray']

        # NOTE: Version '2.5'
        print("Calculating daily solar insolation (slice method)...")
        temp_start = time.perf_counter()
        daily_insolation = calc_daily_insolation(points, height, world_radius, current_tilt)
        temp_end = time.perf_counter()
        print(f"Runtime: {temp_end-temp_start:.5f} sec")

        # daily_insolation *= (tsi/360)  # Compared to the NASA data for 1980 this is off by like 20 Watts or so, but the cosine angle is basically perfect.
        if args.save_img:
            export_list["daily_insolation"] = [daily_insolation, 'gray']

        do_annual = True
        if do_annual:
            print("Calculating annual solar insolation (slice method)...")
            temp_start = time.perf_counter()
            annual_insolation = calc_yearly_insolation(points, height, world_radius, axial_tilt, snapshot_climate)
            temp_end = time.perf_counter()
            print(f"Runtime: {temp_end-temp_start:.5f} sec")

            if args.save_img:
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

        surface_points["purple"].append(llpoint)
        surface_points["magenta"].extend([points[v] for v in llneighbors])

        print("neighbors:", llneighbors)
        print("neighbor distances:", lldistances)
        print("neighbor xyz:", points[llneighbors[0]], points[llneighbors[1]], points[llneighbors[2]])
        # print(surface_points)
        # print(np.asarray(surface_points["purple"]))
        # print(np.asarray(surface_points["magenta"]))

# Save the world map to a texture file
# =============================================
# ToDo: Something to investigate later is that it might be better to save each image as soon as its source data (height, temperature, etc.) has been calculated
# rather than waiting to do all of them at the end (possible RAM use considerations).
    if args.save_img:
        print("Saving world map image(s)...")

        # ToDo: 16-bit images.
        # For grayscale exports* this has already been added to Nixis but for RGB this may require abandoning the pillow library. https://github.com/python-pillow/Pillow/issues/1888
        # *e.g. elevations rescale 0-65535 with 32768 as sea level (but when heights are already absolute just add 32768)
        # Idea: We could get half-meter precision if we have an export mode where we export bathymetry separately from land elevations;
        # multiply the input array * 2 before rounding to integers, I think. Elevation starts at 0 and goes up, bathymetry starts at 65535 and goes down.

        pixel_data = build_image_data(export_list)
        export_list = None

        save_start = time.perf_counter()
        # Performance:
        # Three 2k * 1k images:   1.15 seconds
        # Three 4k * 2K images:   4.7 seconds
        # Three 8k * 4k images:   18 seconds
        # Three 16k * 8k images:  40 seconds
        # Three 32k * 16k images: 31 seconds
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

    if not args.novis:

        print("Preparing visualization...")
        # The array that we want to use for the scalar bar.
        scalars = {"s-mode":"temperature", "scalars":height}
        # Possible s-modes to control the gradient used by the scalar bar:
            # topography (solid color for ocean, colors for land)
            # bathymetry (colors for ocean height, solid color for land)
            # ocean (some other ocean metric colors, maybe currents, solid color for land)
            # elevation (gradient has a hard break at the shore from ocean to land)
            # temperature and insolation (which is uniform for the full surface from min to max without weird breaks at the land/ocean border)

        # Reshaping the heights to match the shape of the vertices array so we can multiply the verticies * the heights.
        scale_size = 0.1  # Exaggerate the scale to be visible to the naked eye from orbit
        if scale_size == 0:  # Flatten everything
            scale_factor = 0
        elif scale_size == 1:  # Real scale
            scale_factor = 1
        else:  # Use exaggerated scale
            scale_factor = world_radius / max_alt * scale_size
            print("  Exaggerated terrain scale factor is", scale_factor)
#        points *= np.reshape(height/world_radius + 1, (len(points), 1))  # Rather, do it this way instead. NOTE: This should really be done inside the visualize function.
        # https://gamedev.net/forums/topic/316602-increase-magnitude-of-a-vector-by-a-value/3029750/
#        points *= np.reshape((height-ocean_level)/world_radius + 1, (len(points), 1))  # Rather, do it this way instead. NOTE: This should really be done inside the visualize function.
        points *= np.reshape(height * scale_factor/world_radius + 1, (len(points), 1))

        # print("After points")
        # print(points)

        # ToDo: Get user confirmation for large meshes (large in the sense of vertex count but possibly also in radius)
        if args.save_mesh:
            if len(points) < 1500000:
                save_mesh(points, cells, cfg.SAVE_DIR, world_name + '_final')
            # else: # get user confirmation

        visualize(points, cells, height, scalars, zero_level=ocean_level, surf_points=surface_points, radius=world_radius, tilt=current_tilt)

        # ToDo: PyVista puts execution 'on hold' while it visualizes. After the user closes it execution resumes.
        # Consider asking the user right here if they want to save out the result as a png/mesh/point cloud.
        # (Only if those options weren't passed as arguments at the beginning, else do that arg automatically and don't ask)
        # (There's also a PyVista mode that runs in the background and can be repeatedly updated? Look into that.)

# Cleanup
# =============================================
    # temp_path = os.path.join(cfg.SAVE_DIR, world_name + '_config.' + options["settings_format"])
    # if os.path.exists(temp_path):
    #     os.remove(temp_path)


# =============================================


if __name__ == '__main__':
    main()

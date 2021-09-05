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
from climate import assign_surface_temp
# pylint: disable=not-an-iterable

# ToDo List:
# - In the future when we have climate simulation, run a test to see if it's faster to lookup xyz2latlon or to simply store latlon for each vertex in a big master array.
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
# - Find out which takes up more bytes in RAM: color represented as RGB, HSV, or hex value.
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
    parser.add_argument("-n", "--name", 
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
    parser.add_argument("--mesh", action="store_true", 
        help="Export the world as a 3D mesh file.")
    parser.add_argument("--pointcloud", action="store_true", 
        help="Export the world as a 3D point cloud file.")
    parser.add_argument("--database", action="store_true", 
        help="Export the world as a sqlite file.")
    parser.add_argument("--png", action="store_true", 
        help="Save a png map of the world.")

    args = parser.parse_args()

    if args.divisions <= 0:
        print("\n" + "Divisions must be an even, positive integer." + "\n")
        sys.exit(0)
    if args.divisions > 2250:
        print("Hold up, friend. Are you sure your computer can handle such a big world?")
        # ToDo: Get user Y/N confirmation and exit or continue. (and better prompt text that explains it will be (k^2 * 10) + 2 vertices (k^2 * 20 triangles))
        sys.exit(0)
    if (args.divisions % 2) == 0:
        divisions = args.divisions
    else:
        divisions = args.divisions + 1
        print(f"Even numbers only for divisions, please. Setting divisions to {divisions}.")

    if args.seed:
        world_seed = args.seed
        print("World seed:", args.seed)
    else:
        seed_rng = np.random.default_rng()
        world_seed = int(seed_rng.integers(low=0, high=999999))
        print("World seed:", world_seed)
        # rng = np.random.default_rng(i)
        # rng.random()

    if args.name:
        world_name = args.world_name
    else:
        world_name = str(world_seed) + "_" + str(divisions)

    if args.radius:  # ToDo: I wonder if it's proper to do like "world_radius = args.radius or 1.0" to save a few lines of code
        world_radius = args.radius
    else:  # NOTE: Technically speaking I should do the same as I did for divisions and simply set a default in argparse and ot have this if/else statement at all.
        # world_radius = 6378100  # Actual earth radius in meters
        world_radius = 1.0  # In meters

    my_dir = os.path.dirname(os.path.abspath(__file__))
    with open("options.json", "rt") as f:  # ToDo: Handle error if the file does not exist
        options = json.load(f)
    img_width = options["img_width"]
    img_height = options["img_height"]
    save_dir = os.path.join(my_dir, options["save_folder"])
    export_list = options["export_list"]

    KDT = None
    test_latlon = False

    do_erode = False
    do_climate = False

    snapshot_erosion = False
    snapshot_climate = False

    # ToDo: We should possibly only try making the dir at the moment we save.
    try:  # ToDo: Test if the directory already exists. Maybe even attempt to see if we have write permission beforehand.
        os.mkdir(save_dir)
    except:
        # ToDo: Actual exception types
        # print("Failed to create script output directory!")
        pass

    # Save the world preset  # ToDo: These files fill up the directory fast. Maybe only save if user passes an arg to save.
    world_preset = {
        "world_name": world_name,
        "world_seed": world_seed,
        "divisions": divisions,
        "world_radius": world_radius
    }

    save_settings(world_preset, save_dir, world_name + '_settings', fmt=options["settings_format"])

# Start the party
# =============================================
    now = time.localtime()
    print("==============================")
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


# Sample noise and apply to mesh
# ToDo: Sampling noise should only return the simple array of the height values. Multiplying the height * the vertices should be left to the function that needs the heights or right before that function.
# e.g. we sample the heights and return those, and then we Do Stuff with the height (e.g. erosion) and then only before the visualize step do we multiply the heights * verts.
# Because we need the heights by themselves (not multiplied) for a lot of other steps like erosion, calculating wind, altitude temperature, exporting the png maps, etc..
# Also we need to sample height more than one time (octaves) and multiply those together before we ever need to multiply by the vertices. Multiplied is only needed for visualizing a 'finalized_height'.
# =============================================
    # Initialize the permutation arrays to be used in noise generation
    time_start = time.perf_counter()
    perm, pgi = osi.init(world_seed)
    time_end = time.perf_counter()
    print(f"Time to init permutation arrays:   {time_end - time_start :.5f} sec")

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
    # For reference, Mt. Everest is ~8848.86 m above sea level, the lowest land is the Dead Sea at -428 m below sea level, and the Mariana Trench has a depth of at least -10984 m below sea level.
    print("Sampling noise...")

    height = sample_octaves(points, None, perm, pgi, n_octaves, n_init_rough, n_init_strength, n_roughness, n_persistence, ocean_percent, world_radius)  # ToDo: Consider pulling these values from seed json file cuz this line is a long boi now.

#    height2 = sample_octaves(points, height, perm, pgi, n_octaves, 2, 0.25, 0.5, 0.6)

#    height3 = sample_octaves(points, height2, perm, pgi, 2, 1.5, 0.5, 0.25, 0.25)

#    height = sample_noise4(points, perm, pgi, n_init_rough, n_init_strength, world_radius)

    # ha_start = time.perf_counter()
    # heighta = sample_noise4(points, perm, pgi, 1.5 / world_radius, 0.4 / world_radius, world_radius)  # The strength value is straight multiplication, not a power, so by itself it has no effect on the noise; the effect of the strength is during addition of octaves.
    # hb_start = time.perf_counter()
    # heightb = sample_noise4(points, perm, pgi, 3.75 / world_radius, 0.2 / world_radius, world_radius)
    # hc_start = time.perf_counter()
    # heightc = sample_noise4(points, perm, pgi, 9.375 / world_radius, 0.1 / world_radius, world_radius)
    # h_end = time.perf_counter()
    # heightd = sample_noise4(points, perm, pgi, 23.4375 / world_radius, 0.05 / world_radius, world_radius)
    # heighte = sample_noise4(points, perm, pgi, 58.59375 / world_radius, 0.025 / world_radius, world_radius)
    # heightf = sample_noise4(points, perm, pgi, 146.484375 / world_radius, 0.0125 / world_radius, world_radius)
    # heightg = sample_noise4(points, perm, pgi, 366.2109375 / world_radius, 0.00625 / world_radius, world_radius)
    # print(f"Total noise function runtime:   {hb_start - ha_start :.5f} sec")
    # print(f"Total noise function runtime:   {hc_start - hb_start :.5f} sec")
    # print(f"Total noise function runtime:   {h_end - hc_start :.5f} sec")

    # height = (heighta + heightb + heightc)# * 0.4  # Multiplying by a non-power strength number appears to be irrelevant when we do a rescale afterward. The rescale determines the heights.
    #height = heighta
#    height = (heighta + heightb + heightc + heightd + heighte + heightf + heightg)

    height = rescale(height, -0.05, 0.15)
#    height = rescale(height, -4000, 8850)
#    height = rescale(height, -4000, 0, 0, mode='lower')
#    height = rescale(height, 0, 8850, 0, mode='upper')

    minval = np.amin(height)
    maxval = np.amax(height)
    # https://math.stackexchange.com/questions/2110160/find-percentage-value-between-2-numbers
    # ocean_level = minval + ((maxval - minval) * 50 / 100)  # If I want to input percentage as an integer between 0 and 100 we need the divide by 100 step
    ocean_level = minval + ((maxval - minval) * ocean_percent)  # Or this way to just input a decimal percent. NOTE: Due to the way np.clip works, values less than 0% or greater than 100% have no effect.

    # height *= 10
    # height += 6378100
    # height += 400000

    height = np.clip(height, ocean_level, maxval)

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
            erode_terrain3(points, neighbors, height, 11, snapshot=img_query_data)
        else:
            erode_terrain3(points, neighbors, height, 11, snapshot=None)
        erode_end = time.perf_counter()
        print(f"Erosion runtime: {erode_end-erode_start:.5f}")

        # print(newheight)

        if args.png:
            export_list["height"] = height

# Climate Stuff
# =============================================
    if do_climate:
        print("Assigning starting temperatures...")
        temp_start = time.perf_counter()
        if snapshot_climate:
            surface_temps = assign_surface_temp(points, height)  # ToDo: When climate sim actually has steps, implement snapshotting
        else:
            surface_temps = assign_surface_temp(points, height)
        temp_end = time.perf_counter()
        print(f"Assign surface temps runtime: {temp_end-temp_start:.5f} sec")

        if args.png:
            export_list["surface_temp"] = surface_temps


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
    print("Preparing visualization...")
#    visualize(noise_vals, cells, height, search_point=llpoint, neighbors=neighbors)
    visualize(points * (np.reshape(height, (len(points), 1)) + 1), cells, height)  # NOTE: Adding +1 to height means values only grow outward if range is 0-1. But for absolute meter ranges it merely throws the values off by 1.

    # ToDo: PyVista puts execution 'on hold' while it visualizes. After the user closes it execution resumes.
    # Consider asking the user right here if they want to save out the result as a png/mesh/point cloud.
    # (Only if those options weren't passed as arguments at the beginning, else do that arg automatically and don't ask)
    # (There's also a PyVista mode that runs in the background and can be repeatedly updated? Look into that.)


@njit(cache=True, parallel=True, nogil=True)
def sample_noise4(verts, perm, pgi, n_roughness=1, n_strength=0.2, radius=1):
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
        print(f"  Octave {i+1}..")
        # sample_noiseX(verts, elevations, perm, pgi, n_freq, n_amp)
        elevations += sample_noise4(verts, perm, pgi, n_freq / world_radius, n_amp / world_radius, world_radius)

        n_freq *= n_roughness
        n_amp *= n_persistence

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


@njit(cache=True, parallel=True, nogil=True)
def sample_noiseX(verts, elevations, perm, pgi, n_roughness=1, n_amp=1.0):
    """Sample a simplex noise for given vertices"""
    rough_verts = verts * n_roughness

#    for v, vert in enumerate(rough_verts):
#        elevations[v] += (osi.noise3d(vert[0], vert[1], vert[2], perm, pgi) + 1) * 0.5 * n_amp
    for v in prange(len(rough_verts)):
        elevations[v] += (osi.noise3d(rough_verts[v][0], rough_verts[v][1], rough_verts[v][2], perm, pgi) + 1) * 0.5 * n_amp

#    rough_verts = None

#    return elevations


def visualize(verts, tris, heights=None, search_point=None, neighbors=None):
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
    pl.add_mesh(mesh, show_edges=False, smooth_shading=True, color="white", below_color="blue", scalars=heights, cmap="algae_r", culling = "back", scalar_bar_args=sargs)
    # pl.add_scalar_bar(below_label="Ocean")
    if search_point is not None and neighbors is not None:
        pl.add_mesh(neighbor_dots, point_size=15.0, color = "magenta")
        pl.add_mesh(search_dot, point_size=15.0, color = "purple")
    pl.add_mesh(x_axisline, line_width=5, color = "red")
    pl.add_mesh(y_axisline, line_width=5, color = "green")
    pl.add_mesh(z_axisline, line_width=5, color = "blue")
    pl.show_axes()
    print("Sending to PyVista.")
    pl.show()

# =============================================


if __name__ == '__main__':
    main()

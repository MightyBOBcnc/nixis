"""A program to generate detailed maps for spherical worlds."""

import os
import sys
import argparse
import time
import math
import concurrent.futures
import meshio as mi
import meshzoo as mz
import numpy as np
from numba import jit, njit
import pyvista as pv
from opensimplex import OpenSimplex
from scipy.spatial import KDTree
from PIL import Image
from util import *

# ToDo List:
# - In the future when we have climate simulation, run a test to see if it's faster to lookup xyz2latlon or to simply store latlon for each vertex in a big master array.
# - See if there is any sort of pattern in the vertex order from meshzoo that would allow a LOD-like system or a way to know every other vertex or every N vertices so
#   that we could use a sparser or lower-fidelity version of the dense sphere for some calculations. (e.g. I'm thinking of collision detection between drifting continents.)
#
#   Pattern notes (these are only applicable in Blender when you DO NOT preserve the vertex order):
#   v0 is always v0.
#   v1 becomes v3 at k=2, and after that you add +4 for every even number of subdivisions.
#   v2 becomes v5 at k=2, then you add +9, then +13, then +17, then +21 etc (the amount you add by grows by 4)
#   v3 becomes v8 at k=2, then you add +16 then +24 then +32, then +40 etc (the amount you add grows by 8)
#   v4 becomes v11 at k=2 then you add +23 then +35 then +47, then +59 etc (the amount you add grows by 12)
#
# - More powerful performance profiling/benchmarking with cprofile
#   https://docs.python.org/3/library/profile.html
#   https://www.machinelearningplus.com/python/cprofile-how-to-profile-your-python-code
#   Or maybe timeit https://docs.python.org/3/library/timeit.html
#   https://www.geeksforgeeks.org/timeit-python-examples
# - Note: Voroni tesselation library: https://github.com/gdmcbain/voropy
# - Idea: Meshzoo is already really fast; I wonder if it can be @njitted

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
    parser.add_argument("--mesh", action="store_true", 
        help="Export the world as a 3D mesh file.")
    parser.add_argument("--pointcloud", action="store_true", 
        help="Export the world as a 3D point cloud file.")
    parser.add_argument("--database", action="store_true", 
        help="Export the world as a sqlite file.")
    parser.add_argument("--png", action="store_true", 
        help="Save a png map of the world.")
    # ToDo: User-specified image dimensions, but only if --png arg is present.
    # Might be doable with groups?
    args = parser.parse_args()

    if args.divisions <= 0:
        print("")
        print("Divisions must be a positive number.")
        sys.exit(0)
    if args.divisions > 1000:
        print("Hold up, friend. Are you sure your computer can handle such a big world?")
        # ToDo: Get user Y/N confirmation and exit or continue. (and better prompt text)
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
        world_seed = seed_rng.integers(low=0, high=999999)
        print("World seed:", world_seed)
        # rng = np.random.default_rng(i)
        # rng.random()

    if args.name:
        world_name = args.world_name
    else:
        world_name = str(world_seed) + "_" + str(divisions)

    planet_radius = 1
    KDT = None
    test_latlon = False

    # ToDo: We should only try making the dir at the moment we save.
    my_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(my_dir, 'output')
    try:
        os.mkdir(out_dir)
    except:
        # ToDo: Actual exception types
        # print("Failed to create script output directory!")
        pass

# Start the party
# =============================================
    now = time.localtime()
    print("==============================")
    print(f"Script started at: {now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}")

    points, cells = create_mesh(divisions)
    if planet_radius != 1:
        points *= planet_radius

    # print("Points:")
    # print(points)
    # print("Cells:")
    # print(cells)

    # If I need to output several meshes for examination
    # for i in range(2, 26, 2):
    #     a, b = create_mesh(i)
    #     save_mesh(a, b, out_dir, f"{i:02}" + '_smooth')

# Saving the mesh here just writes a plain sphere with no elevation offsets
    if args.mesh:
        save_mesh(points, cells, out_dir, world_name + '_smooth')

# Sample noise and apply to mesh
# ToDo: Sampling noise should only return the simple array of the height values. Multiplying the height * the vertices should be left to the function that needs the heights or right before that function.
# e.g. we sample the heights and return those, and then we Do Stuff with the height (e.g. erosion) and then only before the visualize step do we multiply the heights * verts.
# Because we need the heights by themselves (not multiplied) for a lot of other steps like erosion, calculating wind, altitude temperature, exporting the png maps, etc..
# Also we need to sample height more than one time (octaves) and multiply those together before we ever need to multiply by the vertices. Multiplied is only needed for visualizing a 'finalized_height'.
# =============================================
    # Ocean altitude is a *relative* percent of the max altitude
    alt_ocean = 0.4
    n_strength = 0.2
    n_roughness = 1.5
    n_persistence = 0.85
    n_octaves = 3
    # Numba performance when k=900 - SN4: 14.2s, SN3: 16s, SN2: 16.6s, SN1: 19.5s
    print("Sampling noise...")

#    height = sample_octaves(points, None, world_seed, 2, n_roughness, n_strength, n_persistence, alt_ocean)

#    height2 = sample_octaves(points, height, world_seed, n_octaves, 2, 0.25, 0.5, 0.6)

#    height3 = sample_octaves(points, height2, world_seed+1, 2, 1.5, 0.5, 0.25, 0.25)

#    height = samplenoise4(points, world_seed, n_roughness, n_strength)

    heighta = samplenoise4(points, world_seed, 1.6, 0.4)
    heightb = samplenoise4(points, world_seed, 5, 0.2)
    heightc = samplenoise4(points, world_seed, 24, 0.02)

    heightp = (heighta + heightb + heightc) * 0.4

    minval = np.amin(heightp)
    maxval = np.amax(heightp)
    ocean_percent = 0.5 * maxval

    height = np.clip(heightp, ocean_percent, maxval)

    print("min:", minval)
    print("max:", maxval)

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
        if KDT is None:
            KDT = build_KDTree(points)
        print("Saving world map image...")
        # print("before heights:", height)
        # print("flattened arr: ", height.flatten())
        after_h = rescale(height, 0, 255)
        flatter = after_h.flatten()
        # print(flatter)
        # print("after heights: ", np.round(after_h.astype(int)))
        # ToDo: Is flatting even required? If so, what's the performance hit for big meshes?
        pixels = construct_map_export(1024, 512, KDT, flatter.astype(int))
        save_image(pixels, out_dir, world_name)

# Visualize the final planet
# =============================================
    print("Preparing visualization...")
#    visualize(noise_vals, cells, height, search_point=llpoint, neighbors=neighbors)
    visualize(points * (height + 1), cells, height)

    # ToDo: PyVista puts execution 'on hold' while it visualizes. After the user closes it execution resumes.
    # Consider asking the user right here if they want to save out the result as a png/mesh/point cloud.
    # (Only if those options weren't passed as arguments at the beginning, else do that arg automatically and don't ask)
    # (There's also a PyVista mode that runs in the background and can be repeatedly updated, look into that.)

# @njit  # Doesn't work because of the OpenSimplex object
def samplenoise1(verts, world_seed=0, n_roughness=1, n_strength=0.2):
    """Sample a simplex noise for given vertices"""
    time_start = time.perf_counter()

    tmp = OpenSimplex(seed=world_seed)
    elevations = np.ones((len(verts), 1))

    time_el = time.perf_counter()

    # for v in range(len(verts)):
        # x, y, z = verts[v][0] * n_roughness, verts[v][1] * n_roughness, verts[v][2] * n_roughness
        # point = tmp.noise3d(x,y,z) * n_strength + 1
    for v, vert in enumerate(verts):
        x, y, z = vert[0] * n_roughness, vert[1] * n_roughness, vert[2] * n_roughness
        elevations[v] = tmp.noise3d(x,y,z) * n_strength + 1

    time_end = time.perf_counter()

    print(f"Time to initialize np.ones:     {time_el - time_start :.5f} sec")
    print(f"Time to sample noise:           {time_end - time_el :.5f} sec")
    print(f"Total noise function runtime:   {time_end - time_start :.5f} sec")

    # print("Elevations:")
    # print(elevations)
    return elevations

def samplenoise2(verts, world_seed=0, n_roughness=1, n_strength=0.2):
    time_start = time.perf_counter()

    tmp = OpenSimplex(seed=world_seed)

    rough_verts = verts * n_roughness
    elevations = np.array([[tmp.noise3d(v[0],v[1],v[2]) * n_strength + 1] for v in rough_verts])

    # value = noise.pnoise2(noise_x, noise_y, octaves, persistence, lacunarity, random.randint(1, 9999))

    # rand_seed = np.ones((noise_x.size, ))   # just for comparison
    # value = np.array([noise.pnoise2(x, y, octaves, persistence, lacunarity, r) for x, y, r in zip(noise_x.flat, noise_y.flat, rand_seed)])
    # return value.reshape(noise_x.shape)

    time_end = time.perf_counter()

    print(f"Total noise function runtime:   {time_end - time_start :.5f} sec")
    return elevations

def samplenoise3(verts, world_seed=0, n_roughness=1, n_strength=0.2):
    time_start = time.perf_counter()
    tmp = OpenSimplex(seed=world_seed)

    rough_verts = verts * n_roughness
    elevations = np.array([[tmp.noise3d(v[0],v[1],v[2])] for v in rough_verts]) * n_strength + 1

    time_end = time.perf_counter()

    print(f"Total noise function runtime:   {time_end - time_start :.5f} sec")
    return elevations

def samplenoise4(verts, world_seed=0, n_roughness=1, n_strength=0.2):
    """Sample a simplex noise for given vertices"""
    time_start = time.perf_counter()
    tmp = OpenSimplex(seed=world_seed)
    elevations = np.ones(len(verts))

    time_el = time.perf_counter()

    # It's faster to make another pre-multiplied array than it is to do:
    # tmp.noise3d(vert[0]*n_roughness, vert[1]*n_roughness, vert[2]*n_roughness)
    # For k=900 it only takes 0.1s to multiply, but 5.0s the other way.
    rough_verts = verts * n_roughness
    time_noise = time.perf_counter()

    for v, vert in enumerate(rough_verts):
        elevations[v] = tmp.noise3d(vert[0],vert[1],vert[2])

    time_end = time.perf_counter()

    print(f"Time to initialize np.ones:     {time_el - time_start :.5f} sec")
    print(f"Time to multiply n_roughness:     {time_noise - time_el :.5f} sec")
    print(f"Time to sample noise:           {time_end - time_noise :.5f} sec")
    print(f"Total noise function runtime:   {time_end - time_start :.5f} sec")
    return np.reshape((elevations + 1) * 0.5 * n_strength, (len(verts), 1))

def sample_octaves(verts, elevations=None, world_seed=0, n_octaves=1, n_roughness=2, n_strength=0.2, n_persistence=0.5, alt_ocean=0.4):
    if elevations is None:
        elevations = np.ones((len(verts), 1))
    n_freq = 1.0  # Frequency
    n_amp = 1.0  # Amplitude
    n_ocean = alt_ocean / n_octaves

    tmp = OpenSimplex(seed=world_seed)

    for i in range(n_octaves):
        print(f"Octave {i+1}..")
        elevations = elevations + samplenoiseX(verts, elevations, tmp, n_freq, n_amp)

        n_freq *= n_roughness
        n_amp *= n_persistence

    time_nmin = time.perf_counter()
    emin = np.amin(elevations)
    time_nmax = time.perf_counter()
    emax = np.amax(elevations)
    time_end = time.perf_counter()
    print(f"Time to find numpy min: {time_nmax - time_nmin :.5f} sec")
    print(f"Time to find numpy max: {time_end - time_nmax :.5f} sec")
    print("min:", emin)
    print("max:", emax)
    ocean_percent = alt_ocean * emax
    return np.clip(elevations, ocean_percent, emax) * n_strength
    # return elevations * n_strength

def samplenoiseX(verts, elevations, tmp, n_roughness=1, n_amp=1.0):
    """Sample a simplex noise for given vertices"""
    time_start = time.perf_counter()

    # It's faster to make another pre-multiplied array than it is to do:
    # tmp.noise3d(vert[0]*n_roughness, vert[1]*n_roughness, vert[2]*n_roughness)
    # For k=900 it only takes 0.1s to multiply, but 5.0s the other way.
    rough_verts = verts * n_roughness
    time_noise = time.perf_counter()
    for v, vert in enumerate(rough_verts):
        elevations[v] = (tmp.noise3d(vert[0],vert[1],vert[2]) + 1) * 0.5 * n_amp

    time_end = time.perf_counter()
    rough_verts = None

    print(f"Time to multiply n_roughness:     {time_noise - time_start :.5f} sec")
    print(f"Time to sample noise:           {time_end - time_noise :.5f} sec")
    print(f"Total noise function runtime:   {time_end - time_start :.5f} sec")
    return elevations

# Disappointing. Multithreading does improve the sample time but does not
# Combine with numba-fied opensimplex at all. In fact it's WORSE than
# plain numba-fied opensimplex.
def samplenoise5(verts, world_seed=0, n_roughness=1, n_strength=0.2):
    tmp = OpenSimplex(seed=world_seed)
    elevations = np.ones((len(verts), 1))
    rough_verts = verts * n_roughness

    # Observation: When threads=16 and chunks=500 you get about the same runtime
    # as threads=8 and chunks=1000 (~63 seconds)
    # Chunks of 100 doesn't perform well, and chunks of 10000 aren't great either
    num_threads = 8
    chunk_size = 1000

    time_start = time.perf_counter()

    stupid_workaround = np.full((len(verts), 1), tmp)

    time_workaround = time.perf_counter()
    # elevations *= tmp.noise3d([rough_verts[v][0],rough_verts[v][1],rough_verts[v][2] for v in rough_verts]) * n_strength + 1
    #               [[tmp.noise3d(v[0],v[1],v[2])] for v in roughed_verts]

#    potato = make_ranges(len(verts), num_threads)

    #Process the rows in chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(num_threads) as executor:
#        for row, result in executor.map(dumb_function, rough_verts, chunksize=100):
#        executor.map(lambda v: onesample(tmp, n_strength, v), rough_verts)
# =========
        for row, result in executor.map(foursample, zip(rough_verts, stupid_workaround, range(len(rough_verts))), chunksize=chunk_size):
            elevations[row] = result * n_strength + 1
# =========
#        executor.map(foursample, zip(rough_verts, stupid_workaround, range(len(rough_verts)), elevations), chunksize=chunk_size)

    # for v, vert in enumerate(rough_verts):
    #     onesample(elevations, tmp, v, vert, n_strength)

    time_end = time.perf_counter()
    print(f"Stupid workaround runtime:   {time_workaround - time_start :.5f} sec")
    print(f"Total noise function runtime:   {time_end - time_start :.5f} sec")
    return elevations

def onesample(noise_object, n_strength, vert):
    value = noise_object.noise3d(vert[0],vert[1],vert[2]) * n_strength + 1
#    print("VALUE IS THIS THING:", value)
    return value

def twosample(arr, noise_object, n_strength, v):
    value = noise_object.noise3d(arr[v][0],arr[v][1],arr[v][2]) * n_strength + 1
#    print("VALUE IS THIS THING:", value)
    return v, value

def threesample(arr, noise_object, n_strength, vert):
    index = arr.index(vert)
    print("index:",index)
    value = noise_object.noise3d(vert[0],vert[1],vert[2]) * n_strength + 1
#    print("VALUE IS THIS THING:", value)
    return index, value

def foursample(x):
    index = x[2]
    value = x[1][0].noise3d(x[0][0],x[0][1],x[0][2])
#    x[3][index] = value * 0.05 + 1
    return index, value

def dumb_function(i):
    print("i is", i)
    return 5

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
    print(f"Time to resize triangle array: {time_end - time_start :.5f} sec")

    # pyvista mesh
    mesh = pv.PolyData(verts, new_tris)

    if search_point is not None and neighbors is not None:
        # Is it strictly necessary that these be np.arrays?
        neighbor_dots = pv.PolyData(np.array([verts[neighbors[0]], verts[neighbors[1]], verts[neighbors[2]]]))
        search_dot = pv.PolyData(np.array(search_point))
    x_axisline = pv.Line([-1.5,0,0],[1.5,0,0])
    y_axisline = pv.Line([0,-1.5,0],[0,1.5,0])
    z_axisline = pv.Line([0,0,-1.5],[0,0,1.5])

    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges = False, color = "white", scalars=heights, culling = "back")
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

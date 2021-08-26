"""Some utility functions."""
import os
import sys
import time
import json
import meshzoo as mz
import meshio as mi
from PIL import Image
import numpy as np
from numba import njit, prange
from scipy.spatial import KDTree
# pylint: disable=not-an-iterable


def create_mesh(divisions):
    """Make a mesh with meshzoo"""
    # Note: meshzoo creates its icosahedrons aligned to the world axis.
    # Maya's icosahedron platonic is rotated -58.282 on Y for some reason.
    # Unsurprisingly, the vertex order is also different.
    # meshzoo's winding looks counter-clockwise?
    # Observation: When k is even meshzoo will put a vertex at the N & S pole.
    # Blender's ico has options for point, edge, or face up. Point up always
    # puts a pentagon at the N & S pole. Edge up behaves like meshzoo.
    #
    # Obtain count of components starting from k:
    #   Verts is (k^2 * 10) + 2
    #   Edges is k^2 * 30
    #   Tris is k^2 * 20
    # Obtain count of components starting from verts:
    #   Edges is (verts-2) * 3
    #   Tris is (verts*2) - 4
    # Obtain count of components starting from edges:
    #   Verts is (edges / 3) + 2
    #   Tris is edges / 1.5
    # Obtain count of components starting from tris:
    #   Verts is (tris+4) / 2
    #   Edges is tris * 1.5
    print("Generating the mesh...")
    print(f"k is {divisions}")
    time_start = time.perf_counter()
    points, cells = mz.icosa_sphere(divisions)
    time_end = time.perf_counter()
    print(f"Number of vertices: {points.shape[0]:,}")
    print(f"Number of triangles: {cells.shape[0]:,}")
    print(f"Mesh generated in {time_end - time_start :.5f} sec")
    return points, cells

# ToDo: Experiment with lat/lon performance benchmarking.
# 1. Does performance change if the 180 in latlon2xyz is Int or Float?
# 2. Does performance change if pi/180 is replaced with a constant?
# 3. Is there any speed difference between e.g. np.arcsin / math.arcsin, etc.?
# 4. Test on a large image export: Will latlon2xyz run any faster with @njit?

# https://stackoverflow.com/questions/56945401/converting-xyz-coordinates-to-longitutde-latitude-in-python/56945561#56945561
@njit(cache=True)
def xyz2latlon(x, y, z, r=1):
    """Convert 3D spatial XYZ coordinates into Latitude and Longitude."""
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))

#    lat = math.asin(z / r)
#    lon = math.atan2(y, x)
    return (lat, lon)

# This answer is mostly right but we have to compensate for degrees/radians.
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates/1185413#1185413
@njit(cache=True)
def latlon2xyz(lat, lon, r=1):
    """Convert Latitude and Longitude into 3D spatial XYZ coordinates."""
    x = r * np.cos(lat*(np.pi/180)) * np.cos(lon*(np.pi/180))
    y = r * np.cos(lat*(np.pi/180)) * np.sin(lon*(np.pi/180))
    z = r * np.sin(lat*(np.pi/180))
    return (x, y, z)

# https://stats.stackexchange.com/questions/351696/normalize-an-array-of-numbers-to-specific-range
# https://learn.64bitdragon.com/articles/computer-science/data-processing/min-max-normalization
# ToDo: Need to take a closer look at the shape of the data that is acceptable for this function
# because I've had some weird issues with numba not wanting to compile the return values
# if the array wasn't flattened before being set to rescale()
# ToDo: Check for divide by 0 errors in the logic.
# ToDo: This could possibly be faster with prange.
# ToDo: If "mid" is supplied, then rescale the upper and lower values separately. (rescale with two ranges, lower to mid, and mid to upper)
@njit(cache=True)
def rescale(x, lower, upper, mid=None, mode=None):
    """Re-scale (normalize) a list, x, to a given lower and upper bound."""
    if mode is None:
        if mid is None:
            x_min = np.min(x)
            x_max = np.max(x)
            x_range = x_max - x_min
            new_range = upper - lower
            return np.array([((a - x_min) / x_range) * new_range + lower for a in x])
            # ToDo: should this function even have a return value? Will we ever need the unmodified values? This can just operate on the input x array and replace its values?
            #   Yes, there are times when we want a return value, like when we're rescaling to export a png; we don't want the original data to be modified there.
        else:
            print("Rescale upper and lower values separately.")
    if mode is not None:
        if mid is None:
            print("ERROR: Must supply a middle value to use rescale modes.")  # ToDo: Maybe raise an exception instead
        elif mode == 'lower':
            x_min = np.min(x)
            x_max = np.max(x)
            midval = mid * x_max
            x_range = midval - x_min
            # new_range = upper - lower
            new_range = 0 - lower  # The upper value for rescaling the lower values becomes 0  NOTE: Hold up, I think opensimplex might be returning values in the range 0 to 1, not -1 to +1
            for a in prange(len(x)):
                if x[a] <= midval:
                    x[a] = ((x[a] - x_min) / x_range) * new_range + lower
        elif mode == 'upper':
            print("lol, implement me plz")


def make_ranges(arr_len, threads):
    """Make number ranges for threading from a given length.
    arr_len -- Integer. The length of the data to be split.
    threads -- Integer. The number of chunks/threads you want to use.
    """
    x = int(arr_len/threads)
    # print("x is", x)
    range_list = []
    for i in range(threads + 2):
        if 1 < i < threads + 1:
            range_list.append(((i-1)*x + 1, i*x))
        elif i == 0:
            range_list.append((i, i+x))
        elif i == threads:
            range_list.append(((i-1)*x + 1, arr_len))
    # print(range_list)
    return range_list

def build_KDTree(points, lf=10):
    """Build a KD-Tree from vertices with scipy."""
    print("Building KD Tree...")
    time_start = time.perf_counter()
    # Default leaf size is 10 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    KDT = KDTree(points, leafsize=lf)
    time_end = time.perf_counter()
    print(f"KD built in {time_end - time_start :.5f} sec")
    return KDT

# =============================================
# Image-related utilities
# =============================================

@njit(cache=True, parallel=True, nogil=True)
def make_ll_arr(width, height):
    """Create an array of XYZ coordinates from latitudes and longitudes"""
    map_array = np.ones((height, width, 3), dtype=np.float64)
    lat = 90
    lon = 180
    latpercent = 180 / (height - 1)
    # lonpercent = 360 / (width - 1)
    lonpercent = 360 / width  # ToDo: Fix hairline seam where the image wraps around the sides. We don't actually want to reach -180 because it's the same as +180. (Might not be an issue at higher subdivisions >= 900)

    # Loop through Latitude starting at north pole. Pixels are filled per row.
    for h in prange(height):
        # Loop through each Longitude per latitude.
        for w in prange(width):
            d = latlon2xyz(max(lat - (h * latpercent), -90), max(lon - (w * lonpercent), -180))
            map_array[h][w] = d
    return map_array

# NOTE: Keep an eye out for weird numba compile failures in here.
@njit(cache=True, parallel=True, nogil=True)
def make_m_array(width, height, dists, nbrs, colors):
    """Sample vertices and build a map for export."""
    # debug_color = np.array([255,0,255], dtype=np.int32)
    map_array = np.full((height, width, 3), 255, dtype=np.int32)
    # print(map_array)

    for h in prange(height):
        for w in prange(width):
            # https://math.stackexchange.com/questions/3817854/formula-for-inverse-weighted-average-smaller-value-gets-higher-weight
            sd = np.sum(dists[h][w])  # Sum of distances
            # Add a tiny amount to each i to (hopefully) prevent NaN and div by 0 errors
            ws = np.array([1 / ((i+0.00001) / sd) for i in dists[h][w]], dtype=np.float64)  # weights
            t = np.sum(ws)  # Total sum of weights
            iw = np.array([i/t for i in ws], dtype=np.float64)  # Inverted weights
            value = int(np.sum(np.array([colors[nbrs[h][w][i]]*iw[i] for i in range(len(iw))])))

            map_array[h][w] = [value, value, value]
    return map_array

def build_image_data(imgquery, colors=None):  # Could rename this to like make_image_data
    """Use KD-Tree to sample vertices and build a map for export."""
    # ToDo: In the future I might have a vert array with vertex colors that
    # aren't in a separate height array so I'll need to add a verts argument or
    # something, and an index for which sub-array holds the vertex colors.
    # (This is why colors=None, in case it isn't passed, but hasn't been handled yet..)
    # Also need support for RGB arrays or RGBA if A stores a mask like rivers.
    # Possible improvements include averaging up to 6 neighbors if lat/lon is
    # directly on a vertex.

    with open("options.json", "rt") as f:
        o = f.read()
    options = json.loads(o)
    width = options["img_width"]
    height = options["img_height"]

    print("Sampling verts for texture...")

    dst = imgquery[0]
    nbrs = imgquery[1]
    # print(type(dst))
    # print(type(nbrs))

    time_start = time.perf_counter()
    pixels = make_m_array(width, height, dst, nbrs, colors)
    time_end = time.perf_counter()
    print(f"  Pixels built in   {time_end - time_start :.5f} sec")

    return pixels  # ToDo: For the future, with multiple maps saved at once, maybe return a dict with the map name/type, and the array.

# ToDo: Support for saving images with higher bit depths
# and maybe additional formats (tiff, exr, etc).
def save_image(data, path, name):
    """Save a numpy array as an image file."""
    out_path = os.path.join(path, f"{name}.png")
    img = Image.fromarray(data.astype('uint8'))
    img.save(out_path)

# ToDo: Validate that we can import 16-bit data and keep it that way instead of converting to 8-bit
# Read an image into an array
def image_to_array(image_file):
    """Load an image into a numpy array to use as source heights."""
    img = Image.open(image_file)

    # 8-bit pixels, grayscale (8-bit image)
    if img.mode == "L":
        data = np.asarray(img)
        data = np.float64(data) / 256
    # 32-bit signed integer pixels, grayscale (16-bit image)
    elif img.mode == "I":
        data = np.asarray(img) / 65536
    # 3x8-bit or 4x8-bit pixels or higher, true color (24-bit or higher image)
    elif img.mode in ("RGB", "RGBA"):
        converted = img.convert("L")
        data = np.asarray(converted)
        data = np.float64(data) / 256
    else:
        print("ERROR. Unsupported image format.")
        sys.exit(-1)
    return data

# =============================================

# ToDo: Get user Y/N confirmation for large meshes (might actually do this back in main before calling this)
# - And MAYBE add support for multiple output formats (e.g. ply, obj, etc)
# - Add support for exporting vertex color? But which data to use? Height?
# It should be noted that Blender's obj importer will change the vertex order
# by default and won't match the vert order from meshzoo's points/cells arrays.
# This is not a problem with Blender's ply importer.
# You must select "Keep Vert Order" and Y Forward and Z up to match ply import.
# Maya's default obj importer settings do not have this problem.
# However Maya uses a Y-up coordinate system by default so everything's rotated
# but vert order is properly preserved and rotation or coordinate system can be
# changed within Maya.
def save_mesh(verts, tris, path, name, fmt=None):
    """Save the planet as a 3D mesh that can be used in other software."""
    # meshio mesh
    # mesh = mi.Mesh(points, cells)
    print("Saving mesh to disk...")
    time_start = time.perf_counter()
    out_path = os.path.join(path, f"{name}.obj")
    mi.write_points_cells(out_path, verts, {"triangle": tris})
    time_end = time.perf_counter()
    print(f"Mesh saved in {time_end - time_start :.5f} sec")

# ToDo: Implement this.  ply format, maybe obj? obj supports vertex data only?
# https://info.vercator.com/blog/what-are-the-most-common-3d-point-cloud-file-formats-and-how-to-solve-interoperability-issues
def save_point_cloud(verts, path, name, fmt=None):
    """Save the planet as a point cloud that can be used in other software."""
    print("Not implemented yet.")

def save_settings(data, path, name, fmt=None):
    """Save the variables that will recreate this planet."""
    out_path = os.path.join(path, f"{name}.{fmt}")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

# ToDo: Implement this.
def load_settings(path):
    """Load planet variables from a saved file."""
    print("Not implemented yet.")

# ToDo: Implement this. I'm thinking sqlite if there's a good python library.
# https://sqlite.org/whentouse.html
def export_planet(data, path, name):
    """Export the planet as a database file."""
    print("Not implemented yet.")

# https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast/29799815#29799815
@njit(cache=True)
def find_first(item, vec):
    """Return the index of the first occurence of item in vec."""
    for i in range(len(vec)):  # Seems a smidge faster than enumerate
        if item == vec[i]:
            return i
    # Safety fallback (probably shouldn't ever happen)
    return -1

# ToDo: This is pretty fast for k=2500 but maybe try multithreading later.
# Could make a secondary function that just returns the index position and the
# value and then multithread map that and do the actual setting of the values
# in the adj array in build_adjacency like the make_square test.py example.
# vert = t[0]
# idx = find_first(-1, adj[vert])
# val = t[1]
# return vert, idx, val
@njit(cache=True)
def build_adjacency(triangles):
    """Build an array of adjacencies for each mesh vertex."""
    # The 12 original vertices have connectivity of 5 instead of 6 so for the
    # 6th number we assign -1 as there's no such thing as a vert with ID of -1
    adj = np.full((int((len(triangles)+4)/2), 6), -1, dtype=np.int32)

    for t in triangles:
        # v0 = t[0]
        # v1 = t[1]
        # v2 = t[2]
        # i0 = find_first(-1, adj[v0])
        # i1 = find_first(-1, adj[v1])
        # i2 = find_first(-1, adj[v2])
        # adj[v0][i0] = v1
        # adj[v1][i1] = v2
        # adj[v2][i2] = v0

        adj[t[0]][find_first(-1, adj[t[0]])] = t[1]
        adj[t[1]][find_first(-1, adj[t[1]])] = t[2]
        adj[t[2]][find_first(-1, adj[t[2]])] = t[0]
    # print(triangles)
    # print(adj)
    return adj
    # (Note that this data shape wouldn't have a 'cycle' like delaunator so I couldn't 'walk' forward or backward for, e.g. limiting the allowed output edges for river networks to prevent ugly loopbacks)
    #
    # There is also a nice numpy way to slice up the tris array, vertically, to build some tuple pairs.
    # Maybe it would be handy to build both array types, one of tuples and the other that knows every vert's neighbors for fast reference or something.
    # Also due to the existence of the winding order the tuple slicing wouldn't have to be run forward and then backward, because the neighbor face of an edge already 'runs it backward'.
    # e.g. The triangles [0,11,5] and [0,10,11] would produce [0-->11] and [11-->0] pairs when sliced forward only [v0,v1][v1,v2][v2,v0]
    # However we could probably build any tuples required on the fly like (v, adj[v][x]) where v is the first vert ID and x is the other vert ID from inside adj[v]

# ~10x faster than a list comprehension when numba gets its hands on it
@njit(cache=True)
def next_vert(v, arr1, arr2):
    for i in arr1:
        if i in arr2:
            if i != v:
                return i
    return -1

# Using numba's prange instead of python's range lets numba auto-parallelize
# Takes about 5 seconds for a k=2500 mesh, otherwise would take ~10 seconds.
# Note: This produces a proper 'cycle' for each vertex but the winding order
# is not consistent; some are clockwise and some are counter-clockwise.
@njit(cache=True, parallel=True, nogil=True)
def sort_adjacency(adj):
    for idx in prange(12):
        visited = np.full(6, -1, dtype=np.int32)
        pv = idx
        nv = adj[idx][0]
        # First 12 verts have connectivity 5
        for i in prange(4):
            visited[i] = nv
            nv = next_vert(pv, adj[idx], adj[nv])
            pv = visited[i]
        visited[4] = nv
        adj[idx] = visited

    for idx in prange(12, len(adj)):
        visited = np.full(6, -1, dtype=np.int32)
        pv = idx
        nv = adj[idx][0]
        # All remaining verts have connectivity 6
        for i in prange(5):
            visited[i] = nv
            nv = next_vert(pv, adj[idx], adj[nv])
            pv = visited[i]
        visited[5] = nv
        adj[idx] = visited

"""Some utility functions."""
import os
import sys
import time
import meshzoo as mz
import meshio as mi
from PIL import Image
import numpy as np
from numba import njit
from scipy.spatial import KDTree


def create_mesh(divisions):
    # Note: meshzoo creates its icosahedrons aligned to the world axis.
    # Maya's icosahedron platonic is rotated -58.282 on Y for some reason.
    # Unsurprisingly, the vertex order is also different.  # ToDo: Was that because of the obj importer?  Check this in maya again.
    # meshzoo's winding looks counter-clockwise?
    # Observation: When k is even meshzoo will put a vertex at the N & S pole.
    # Blender's ico has options for point, edge, or face up. Point up always
    # puts a pentagon at the N & S pole. Edge up behaves like meshzoo.
    print("Generating the mesh...")
    print(f"k is {divisions}")
    time_start = time.perf_counter()
    points, cells = mz.icosa_sphere(divisions)
    time_end = time.perf_counter()
    print(f"Number of vertices: {points.shape[0]:,}")  # Verts is (tris+4) / 2
    print(f"Number of triangles: {cells.shape[0]:,}")  # Tris is (verts*2) - 4
    print(f"Mesh generated in {time_end - time_start :.5f} sec")
    return points, cells

# ToDo: Experiment with lat/lon performance benchmarking.
# 1. Does performance change if the 180 in latlon2xyz is Int or Float?
# 2. Does performance change if pi/180 is replaced with a constant?
# 3. Is there any speed difference between e.g. np.arcsin / math.arcsin, etc.?
# 4. Test on a large image export: Will latlon2xyz run any faster with @njit?

# https://stackoverflow.com/questions/56945401/converting-xyz-coordinates-to-longitutde-latitude-in-python/56945561#56945561
def xyz2latlon(x, y, z, r=1):
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))

#    lat = math.asin(z / r)
#    lon = math.atan2(y, x)
    return (lat, lon)

# This answer is mostly right but we have to compensate for degrees/radians.
# https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates/1185413#1185413
# @njit
def latlon2xyz(lat, lon, r=1):
    x = r * np.cos(lat*(np.pi/180)) * np.cos(lon*(np.pi/180))
    y = r * np.cos(lat*(np.pi/180)) * np.sin(lon*(np.pi/180))
    z = r * np.sin(lat*(np.pi/180))
    return (x, y, z)

# https://stats.stackexchange.com/questions/351696/normalize-an-array-of-numbers-to-specific-range
# https://learn.64bitdragon.com/articles/computer-science/data-processing/min-max-normalization
def rescale(x, lower, upper):
    """Re-scale (normalize) a list, x, to a given lower and upper bound."""
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    new_range = upper - lower
    return np.array([((a - x_min) / x_range) * new_range + lower for a in x])

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

def build_KDTree(points):
    print("Building KD Tree...")
    time_start = time.perf_counter()
    # Default leaf size is 10 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    KDT = KDTree(points)
    time_end = time.perf_counter()
    print(f"KD built in {time_end - time_start :.5f} sec")
    return KDT

# @njit  # Won't work with the tree object. Maybe one day numba-scipy will work
# https://github.com/numba/numba-scipy
def construct_map_export(width, height, tree, colors=None):
    """Use KD-Tree to sample vertices and build a map for export."""
    # ToDo: In the future I might have a vert array with vertex colors that
    # aren't in a separate height array so I'll need to add a verts argument or
    # something, and an index for which sub-array holds the vertex colors.
    # (This is why colors=None, in case it isn't passed, but hasn't been handled yet..)
    # Also need support for RGB arrays or RGBA if A stores a mask like rivers.
    # Possible improvements include averaging up to 6 neighbors if lat/lon is
    # directly on a vertex, and 'weighting' the contribution of each vert
    # to the averaged color based on distance (which we know, thanks to KDtree)

    # Initialize array with magenta as debug color in case pixels get missed
    map_array = np.full((height, width, 3), [255, 0, 255])
    print("Sampling verts for texture...")
    lat = 90
    lon = 180
    latpercent = 180 / (height - 1)
    # lonpercent = 360 / (width - 1)
    lonpercent = 360 / width  # ToDo: Fix hairline seam where the image wraps around the sides. We don't actually want to reach -180 because it's the same as +180. (Might not be an issue at higher subdivisions >= 900)

    time_start = time.perf_counter()
    # ToDo: This is slow. Speed it up. Problem: numba hates the tree object.
    # Maybe somehow I could do an Enumerate instead? As with all of my problems, the hard part is quickly filling the arrays with the data to begin with, not the computations on the data once the arrays are already full.
    # There may also be some sort of benefit to pre-calculating the nearest N neighbors for each lat/lon into their own array in the background?
    # (in the future when there's enough stuff going on that there is even time for backgrond processing)
    # Loop through Latitude starting at north pole. Pixels are filled per row.
    for h in range(height):
#        print(f"Starting row {h+1}")
        # Loop through each Longitude per latitude.
        for w in range(width):
            # Query result[0] is distances, result[1] is vert IDs
            # ToDo: I wonder if there is any notable difference to using 2 vars instead of 1 (like distances, neighbors = yadda instead of neighbors = yadda)
            # ToDo: Query can accept an array of points to query which may be faster than doing one point at a time. It could also then be done outside of the function and passed as an array, which would allow @njit decorating construct_map_export
            # How to get a weighted average for vertex colors:
            # https://math.stackexchange.com/questions/3817854/formula-for-inverse-weighted-average-smaller-value-gets-higher-weight
            d, nbr = tree.query(latlon2xyz(lat,lon), 3)
            sd = sum(d)
            ws = [1/(i/sd) for i in d]
            t = sum(ws)
            u = [i/t for i in ws]  # Inverted weights
            value = int(sum([colors[nbr[i]]*f for i, f in enumerate(u)]))

            map_array[h][w] = [value, value, value]
            # if lat == -90:
            #     print("Lon:", lon)
            lon -= (lonpercent)
            if lon < -180:
                lon = -180
        # print("Lat:", lat)
        lat -= (latpercent)
        if lat < -90:
            lat = -90
        lon = 180  # Reset to 180 for each new pixel row.
    
#    print("The map array is now:", map_array)
    time_end = time.perf_counter()
    print(f"Finished sampling in {time_end - time_start :.5f} sec")
    return map_array

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
def save_mesh(verts, tris, path, name, format=None):
    """Save the planet as a 3D mesh that can be used in other software."""
    # meshio mesh
    # mesh = mi.Mesh(points, cells)
    print("Saving mesh to disk...")
    time_start = time.perf_counter()
    out_path = os.path.join(path, f"{name}.ply")
    mi.write_points_cells(out_path, verts, {"triangle": tris})
    time_end = time.perf_counter()
    print(f"Mesh saved in {time_end - time_start :.5f} sec")

# ToDo: Implement this.  ply format, maybe obj? obj supports vertex data only?
# https://info.vercator.com/blog/what-are-the-most-common-3d-point-cloud-file-formats-and-how-to-solve-interoperability-issues
def save_point_cloud(verts, path, name, format=None):
    """Save the planet as a point cloud that can be used in other software."""
    print("Not implemented yet.")

# ToDo: Implement this. Output as a json or txt file or something.
def save_settings(object, path, name, format=None):
    """Save the variables that will recreate this planet."""
    print("Not implemented yet.")

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
def build_adjacency(triangles):
    """Build an array of adjacencies for each mesh vertex."""
    # The 12 original vertices have connectivity of 5 instead of 6 so for the
    # 6th number we assign -1 as there's no such thing as a vert with ID of -1
    adj = np.full((int((len(triangles)+4)/2), 6), -1, dtype=np.int32)

    for t in triangles:
        # i0 = find_first(-1, adj[t[0]])
        # i1 = find_first(-1, adj[t[1]])
        # i2 = find_first(-1, adj[t[2]])
        # adj[t[0]][i0] = t[1]
        # adj[t[1]][i1] = t[2]
        # adj[t[2]][i2] = t[0]

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

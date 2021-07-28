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
    # Unsurprisingly, the vertex order is also different.
    # meshzoo's winding looks counter-clockwise?
    # Observation: When k is even meshzoo will put a vertex at the N & S pole.
    # Blender's ico has options for point, edge, or face up. Point up always
    # puts a pentagon at the N & S pole. Edge up behaves like meshzoo.
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

def xyz2latlon(x, y, z, r=1):
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))

#    lat = math.asin(z / r)
#    lon = math.atan2(y, x)
    return (lat, lon)

# @njit
def latlon2xyz(lat, lon, r=1):
    x = r * np.cos(lat*(np.pi/180)) * np.cos(lon*(np.pi/180))
    y = r * np.cos(lat*(np.pi/180)) * np.sin(lon*(np.pi/180))
    z = r * np.sin(lat*(np.pi/180))
    return (x, y, z)

# https://learn.64bitdragon.com/articles/computer-science/data-processing/min-max-normalization
def rescale(x, lower, upper):
    """Re-scale (normalize) a list, x, to a given lower and upper bound."""
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    new_range = upper - lower
    return np.array([((a - x_min) / x_range) * new_range + lower for a in x])

def make_ranges(arr_len, threads):
    """Split vert array into ranges for threading.
    arr_len -- Integer. The length of the array to be split.
    threads -- Integer. The number of threads you want to use
    """
    x = int(arr_len/(threads))
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

    map_array = np.full((height, width, 3), [255, 0, 255])
    print("Sampling verts for texture...")
    lat = 90
    lon = 180
    latpercent = 180 / (height - 1)
    lonpercent = 360 / (width - 1)

    time_start = time.perf_counter()
    # ToDo: This is slow. Speed it up. Problem: numba hates the tree object.
    # Loop through Latitude. Pixels are filled per row.
    for h in range(height):
        print(f"Starting row {h+1}")
        # Loop through each Longitude per latitude.
        for w in range(width):
            neighbors = tree.query(latlon2xyz(lat,lon), 3)
            value = int((colors[neighbors[1][0]] + colors[neighbors[1][1]] + colors[neighbors[1][2]]) / len(neighbors[1]))
            map_array[h][w] = [value, value, value]
            lon -= (lonpercent)  # ToDo: If we're on the final loop, don't allow lon to exceed -180
        lat -= (latpercent)  # ToDo: If we're on the final loop, don't allow lat to exceed -90
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
# And MAYBE add support for multiple output formats (e.g. ply, obj, etc)
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

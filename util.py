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
import cfg
# pylint: disable=not-an-iterable
# pylint: disable=line-too-long


def create_mesh(divisions):
    """Make a mesh with meshzoo"""
    # NOTE: meshzoo creates its icosahedrons aligned to the world axis.
    # Maya's icosahedron platonic is rotated -58.282 on Y for some reason.
    # Unsurprisingly, the vertex order is also different.
    # meshzoo's winding looks counter-clockwise?
    # Observation: When k is even meshzoo will put a vertex at the N & S pole.
    # Blender's ico has options for point, edge, or face up. Point up always
    # puts a pentagon at the N & S pole. Edge up behaves like meshzoo.
    #
    # Obtain count of components starting from divisions:
    #   Verts is (k^2 * 10) + 2
    #   Edges is k^2 * 30
    #   Tris is k^2 * 20
    # Obtain count of components starting from verts:
    #   Edges is (verts - 2) * 3
    #   Tris is (verts * 2) - 4
    # Obtain count of components starting from edges:
    #   Verts is (edges / 3) + 2
    #   Tris is edges / 1.5
    # Obtain count of components starting from tris:
    #   Verts is (tris + 4) / 2
    #   Edges is tris * 1.5
    print("Generating the mesh...")
    print(f"k is {divisions}")
    time_start = time.perf_counter()
    points, cells = mz.icosa_sphere(divisions)
    time_end = time.perf_counter()
    print(f"Number of vertices: {points.shape[0]:,}")  # points: np.float64
    print(f"Number of triangles: {cells.shape[0]:,}")  # cells: np.int32
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
    # Old method; will output NaNs if (Z/R) > 1 or (Z/R) < -1
    # lat = np.degrees(np.arcsin(z / r))

    # Constrain the value to -1 to 1 before doing arcsin
    lat = np.degrees(np.arcsin(min(max((z / r), -1), 1)))
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

@njit(cache=True)
def kelvin_to_c(k):
    """Convert temperature from Kelvin to Celsius."""
    return k - 273.15

@njit(cache=True)
def c_to_kelvin(c):
    """Convert temperature from Celsius to Kelvin."""
    return c + 273.15

# https://stats.stackexchange.com/questions/351696/normalize-an-array-of-numbers-to-specific-range
# https://learn.64bitdragon.com/articles/computer-science/data-processing/min-max-normalization
# ToDo: Need to take a closer look at the shape of the data that is acceptable for this function
# because I've had some weird issues with numba not wanting to compile the return values
# if the array wasn't flattened before being set to rescale()
# ToDo: Prevent accidental div by 0 errors by checking to see if x_range is 0.
# ToDo: Also NaN checks related to above.
# ToDo: If "mid" is supplied, then rescale the upper and lower values separately. (rescale with two ranges, lower to mid, and mid to upper)
# ToDo: Absolute value scaling support. E.G. If the supplied array is exactly 0 to 1, then each 0.00113019 would scale to approximately 10 meters if our absolute max is Mt Everest at 8848 meters.
# Possible ToDo: A parameter called "inplace" that is either true or false (default false) that modifies the array in place when true.
@njit(cache=True, parallel=True, nogil=True)
def rescale(x, lower, upper, mid=None, mode=None, u_min=None, u_max=None):
    """Re-scale (normalize) an array to a given lower and upper bound.
    x -- The input array.
    lower -- The desired new min value.
    upper -- The desired new max value.
    mid -- Optionally rescale the lower & upper values separately.
    Mid should be between x's existing min and max values.
    mode -- Optionally rescale ONLY the 'lower' or 'upper' values.
    u_min -- Optionally specify an absolute value for x min.
    u_max -- Optionally specify an absolute value for x max.
    """
    new_array = np.copy(x)  # NOTE: Rescale doesn't work safely on certain numpy dtypes, like a simple black and white mask that has a dtype of bool_ or something simple like int8 because their precision is too small.
    # new_array = x.astype(np.float64)

    x_min = np.min(x)
    x_max = np.max(x)

    if u_min is not None and u_min < x_min:
        x_min = u_min
    if u_max is not None and u_max > x_max:
        x_max = u_max

    if mode is None:
        if mid is None:  # Standard rescale. Works fine.
            x_range = x_max - x_min
            new_range = upper - lower

            for a in prange(len(x)):
                new_array[a] = ((x[a] - x_min) / x_range) * new_range + lower
            return new_array

        else:  # Rescale both halves separately. Seems to work fine.
            x_lower_range = mid - x_min
            new_lower_range = mid - lower
            x_upper_range = x_max - mid
            new_upper_range = upper - mid
            for a in prange(len(x)):
                if x[a] <= mid:
                    new_array[a] = ((x[a] - x_min) / x_lower_range) * new_lower_range + lower
                else:
                    new_array[a] = ((x[a] - mid) / x_upper_range) * new_upper_range + mid
            return new_array

    if mode is not None:
        if mid is None:
            print("ERROR: Must supply a middle value to use rescale modes.")
            print("Continuing with unmodified data.")
            # ToDo: Maybe raise an exception instead (actually I don't know if numba can handle that; the documentation says it can't do try/except, but it can do raise or assert)
            # sys.exit(0)
            return x
        elif mode == 'lower':
            x_range = mid - x_min
            new_range = mid - lower  # The upper value for rescaling the lower values becomes 0  NOTE: Hold up, my sample_noise4 is returning values from 0 to 1, not -1 to +1 because of the way I add 1 to values.
            for a in prange(len(x)):
                if x[a] <= mid:
                    new_array[a] = ((x[a] - x_min) / x_range) * new_range + lower
            return new_array

        elif mode == 'upper':
            x_range = x_max - mid
            new_range = upper - mid
            for a in prange(len(x)):
                if x[a] >= mid:
                    new_array[a] = ((x[a] - mid) / x_range) * new_range + mid
            return new_array

# Possible ToDo: A parameter called "inplace" that is either true or false (default false) that modifies the array in place when true.
@njit(cache=True, parallel=True, nogil=True)
def power_rescale(x, mask=None, mode=None, power=1.0):
    """Rescale values using a power function.
    x -- The array to modify.
    mask -- An array of True/False that matches the length of x.
    mode -- 0 means rescale items in x that are False in the mask,
            1 means rescale items in x that are True in the mask.
    power -- The power to apply to items in x."""
    new_array = np.copy(x)

    temp_upper = 1.0
    temp_lower = 0.0

    x_min = np.min(x)
    x_max = np.max(x)

    mask_lower = x_max
    mask_upper = x_min

    temp_range = temp_upper - temp_lower

    # ToDo: We can possibly consolidate the structure in here to shove everything under mode==1 or mode==0 to reduce the number of if/elif branches.
    # There might also be some validity to the idea of building a separate list of vertices to modify and only operating on those instead of doing
    # repeated comparisons of if mode==N and mask[a] for every vertex on the mesh.
    # Also if both rescale and power_rescale will continue to exist as separate functions then there's no reason that power_rescale should do all the math itself for the temporary conversion to 0-1 range and back; just call rescale.
    if mode==1:
        for a in range(len(x)):  # Probably can't safely do prange here
            if mask[a] and x[a] < mask_lower:
                mask_lower = x[a]
            elif mask[a] and x[a] > mask_upper:
                mask_upper = x[a]
    elif mode==0:
        for a in range(len(x)):  # Probably can't safely do prange here
            if not mask[a] and x[a] < mask_lower:
                mask_lower = x[a]
            elif not mask[a] and x[a] > mask_upper:
                mask_upper = x[a]

    print("x min:", x_min)
    print("x max:", x_max)
    print("mask min:", mask_lower)
    print("mask max:", mask_upper)

    mask_range = mask_upper - mask_lower

    # new_array[a] = ((x[a] - x_min) / x_range) * new_range + lower
    for a in prange(len(x)):
        if mode==1 and mask[a]:
            new_array[a] = ((x[a] - mask_lower) / mask_range) * temp_range + temp_lower
        elif mode==0 and not mask[a]:
            new_array[a] = ((x[a] - mask_lower) / mask_range) * temp_range + temp_lower

    for a in prange(len(x)):
        if mode==1 and mask[a]:
            new_array[a] **= power
        elif mode==0 and not mask[a]:
            new_array[a] **= power

    # Attemping to use a fudge factor messes up the range when converting back to the original range?
    # Something to investigate later. But I did accidentally make something that looks like continental shelves.
    # fudge = 1.2
    # temp_lower *= fudge
    # temp_upper *= fudge
    # temp_range = temp_upper - temp_lower
    # for a in prange(len(x)):
    #     if mode==1 and mask[a]:
    #         new_array[a] = (new_array[a] * fudge) ** power
    #     elif mode==0 and not mask[a]:
    #         new_array[a] = (new_array[a] * fudge) ** power

    for a in prange(len(x)):
        if mode==1 and mask[a]:
            new_array[a] = ((new_array[a] - temp_lower) / temp_range) * mask_range + mask_lower
        elif mode==0 and not mask[a]:
            new_array[a] = ((new_array[a] - temp_lower) / temp_range) * mask_range + mask_lower

    return new_array


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
    cfg.KDT = KDTree(points, leafsize=lf)
    time_end = time.perf_counter()
    print(f"KD built in {time_end - time_start :.5f} sec")
    # return cfg.KDT  # No point in returning if it's already saved to cfg

# =============================================
# Image-related utilities
# =============================================

@njit(cache=True, parallel=True, nogil=True)
def make_ll_arr(width, height, radius):
    """Find XYZ coordinates for each pixel's latitude and longitude"""
    # Each pixel that will compose the exported world map has its own lat/lon.
    map_array = np.ones((height, width, 3), dtype=np.float64)
    lat = 90
    lon = -180
    latpercent = 180 / (height - 1)
    # lonpercent = 360 / (width - 1)
    lonpercent = 360 / width  # ToDo: Fix hairline seam where the image wraps around the sides. We don't actually want to reach -180 because it's the same as +180. (Might not be an issue at higher subdivisions >= 900)

    # Loop through Latitude starting at north pole. Pixels are filled per row.
    for h in prange(height):
        # Loop through each Longitude per latitude.
        for w in prange(width):
            # Take the pixel's lat/lon and get its 3D coordinates
            d = latlon2xyz(max(lat - (h * latpercent), -90), max(lon + (w * lonpercent), -180), radius)
            map_array[h][w] = d
    return map_array

# NOTE: Keep an eye out for weird numba compile failures in here.
# ToDo: Want to take a look at refactoring this later without using so many np.array() calls, and possibly without using numpy at all as it may have too much overhead for such small arrays (array size is like 3 items for neighbors)
@njit(cache=True, parallel=True, nogil=True)
def make_rgb_array(width, height, dists, nbrs, colors):
    """Sample vertices and build an RGB map for export."""
    # Had to make RGB and Gray as separate functions because njit barfs
    # if the return shape is different for an RGB/Gray array and because
    # Pillow doesn't understand MxNx1 shaped arrays.
    # debug_color = np.array([255,0,255], dtype=np.int32)
    map_array = np.full((height, width, 3), 0, dtype=np.int32)
    # print(map_array)
    orig_dtype = colors.dtype

    for h in prange(height):
        for w in prange(width):
            # ToDo: For certain maps like a B&W land/water mask, or the tectonic plate map I might actually want hard borders without the anti-aliased smoothing of averages along the edges.
            # For this perhaps I could have the function take an argument like 'AA' for anti-aliashed.  If True, use the weighted distance logic. Else use a different logic. Maybe even split the logics out to different functions.
            # https://math.stackexchange.com/questions/3817854/formula-for-inverse-weighted-average-smaller-value-gets-higher-weight
            sd = np.sum(dists[h][w])  # Sum of distances
            # Add a tiny amount to each i to (hopefully) prevent NaN and div by 0 errors
            ws = np.array([1 / ((i+0.00001) / sd) for i in dists[h][w]], dtype=np.float64)  # weights
            t = np.sum(ws)  # Total sum of weights
            iw = np.array([i/t for i in ws], dtype=np.float64)  # Inverted weights
            value = int(np.sum(np.array([colors[nbrs[h][w][i]]*iw[i] for i in range(len(iw))])))  # ToDo: If I add support for floating point image formats like EXR or HDR then this can't be an int.

            # ToDo: Support for setting channel values individually instead of all the same. Possibly as sub-functions depending on performance and flow/structure.
            map_array[h][w] = [value, value, value]

    return map_array.astype(orig_dtype)

# ToDo: Want to take a look at refactoring this later without using so many np.array() calls, and possibly without using numpy at all as it may have too much overhead for such small arrays (array size is like 3 items for neighbors)
@njit(cache=True, parallel=True, nogil=True)
def make_gray_array(width, height, dists, nbrs, colors):
    """Sample vertices and build a grayscale map for export."""
    # Had to make RGB and Gray as separate functions because njit barfs
    # if the return shape is different for an RGB/Gray array and because
    # Pillow doesn't understand MxNx1 shaped arrays.
    map_array = np.full((height, width), 0, dtype=np.int32)
    # print(map_array)
    orig_dtype = colors.dtype

    for h in prange(height):
        for w in prange(width):
            # ToDo: For certain maps like a B&W land/water mask, or the tectonic plate map I might actually want hard borders without the anti-aliased smoothing of averages along the edges.
            # For this perhaps I could have the function take an argument like 'AA' for anti-aliashed.  If True, use the weighted distance logic. Else use a different logic. Maybe even split the logics out to different functions.
            # https://math.stackexchange.com/questions/3817854/formula-for-inverse-weighted-average-smaller-value-gets-higher-weight
            sd = np.sum(dists[h][w])  # Sum of distances
            # Add a tiny amount to each i to (hopefully) prevent NaN and div by 0 errors
            ws = np.array([1 / ((i+0.00001) / sd) for i in dists[h][w]], dtype=np.float64)  # weights
            t = np.sum(ws)  # Total sum of weights
            iw = np.array([i/t for i in ws], dtype=np.float64)  # Inverted weights
            value = int(np.sum(np.array([colors[nbrs[h][w][i]]*iw[i] for i in range(len(iw))])))  # ToDo: If I add support for floating point image formats like EXR or HDR then this can't be an int.

            map_array[h][w] = value

    return map_array.astype(orig_dtype)

def build_image_data(colors=None):  # Could rename this to like make_image_data (my naming conventions are not unitform, some functions are "make_", some are "build_", and some are "create_"; make_ would be shortest)
    """Use KD-Tree results to sample vertices and build a map for export.
    colors -- A dict of numpy arrays. Keys are names, arrays hold the data.
    """
    # ToDo: In the future I might have a vert array with vertex colors that
    # aren't in a separate height array so I'll need to add a verts argument or
    # something, and an index for which sub-array holds the vertex colors.
    # (This is why colors=None, in case it isn't passed, but hasn't been handled yet..)
    # Also need support for RGB arrays or RGBA if A stores a mask like rivers.
    # Possible improvements include averaging up to 6 neighbors if lat/lon is
    # directly on a vertex. (if dist to 1 vert is <= some tiny number)

    options = load_settings("options.json")  # ToDo: Probably grab options from CFG instead of from disk.
    width = options["img_width"]
    height = options["img_height"]

    print("Sampling verts for texture...")

    dists = cfg.IMG_QUERY_DATA[0]  # numpy float64
    nbrs = cfg.IMG_QUERY_DATA[1]   # numpy int64
    result = {}

    if not isinstance(colors, dict):
        print("ERROR: Must pass a dict when saving out texture maps.")  # ToDo: Better error handling.

    for key, container in colors.items():
        array = container[0]
        mode = container[1]
        if array.dtype in ('int8', 'uint8', 'bool_'):  # uint8 is technically safe for rescale between 0-255
            colors[key] = [rescale(array.astype(np.float64), 0, 255), mode]  # ToDo: Will pillow choke if it gets passed a float64 array?
        elif array.dtype in ('uint16', 'uint32'):  # ToDo: Uh, why did I exempt uint32?
            pass
        else:  # Fallback for unexpected dtypes
            colors[key] = [rescale(array, 0, 255), mode]  # ToDo: Will pillow be okay?  We're not doing an .astype to change the dtype here.

    for key, container in colors.items():
        array = container[0]
        mode = container[1]

        time_start = time.perf_counter()
        if mode in ('rgb', 'RGB'):
            pixels = make_rgb_array(width, height, dists, nbrs, array)
        elif mode in ('gray', 'GRAY', 'grey', 'GREY'):
            pixels = make_gray_array(width, height, dists, nbrs, array)
        time_end = time.perf_counter()

        colors[key] = None

        print(f"  {key} pixels built in   {time_end - time_start :.5f} sec")
        if array.dtype in ('float16', 'int32', 'uint32', 'float32', 'int64', 'uint64', 'float64'):
            result[key] = pixels.astype('uint8')
        else:
            result[key] = pixels.astype(array.dtype)

    return result

# ToDo: Support for saving images with higher bit depths
#    and maybe additional formats (tiff, exr, etc).
# ToDo: Attach metadata to saved images such as:
#   'Generated with Nixis: [link to github]
#   Area per pixel at the equator.
#   The scale of the data, e.g. the min and max height value, or temp, etc.
#   Maybe the basic seed values of seed, divisions, and radius
#   Maybe even steganography for funsies. https://www.thepythoncode.com/article/hide-secret-data-in-images-using-steganography-python
# EXIF and steganography will obviously not work with all image formats, so, research what metadata can be stored on what formats.
# ToDo: Support for user-defined unit scales? (e.g. save temps as C or F? I'm not sure if that would be visually different.)
def save_image(data, path, name):
    """Save a numpy array as an image file.
    data -- A dict. The key will be appended to the end of the file name.
    path -- The path to the folder where the file will be saved.
    name -- File name without extension. Final output will be name_key.ext"""
    # Retrieve file extension from Nixis options.json  # ToDo: Consider using the cfg module instead.
    options = load_settings("options.json")
    fmt = options["img_format"]

    for key, array in data.items():
        out_path = os.path.join(path, f"{name}_{key}.{fmt}")
        # img = Image.fromarray(array.astype('uint8'))
        # img = Image.fromarray(array[:,:,0])  # If an array is RGB (if each pixel has 3 values) this will slice it to only have 1 value, i.e. grayscale, which pillow CAN do.  (Will work for uint16 and uint8 grayscale)
        img = Image.fromarray(array)
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

# ToDo:
# - MAYBE add better support for multiple output formats (e.g. ply, obj, etc)
# - Add support for exporting vertex color? But which data to use? Height?
# - And consider binary vs ASCII formats for file size and import performance.
# It should be noted that Blender's obj importer will change the vertex order
# by default and won't match the vert order from meshzoo's points/cells arrays.
# This is not a problem with Blender's ply importer.
# You must select "Keep Vert Order" and Y Forward and Z up to match ply import.
# Maya's default obj importer settings do not have this problem.
# However Maya uses a Y-up coordinate system by default so everything's rotated
# but vert order is properly preserved and rotation or coordinate system can be
# changed within Maya.
def save_mesh(verts, tris, path, name):
    """Save the planet as a 3D mesh that can be used in other software."""
    if len(tris) > 3000000:  # 3 million is a somewhat arbitrary choice
        print("\n" + f"ATTENTION. This mesh has {len(tris):,} triangles. \
            Your 3D modeling software may not be able to open/edit it." + "\n")
        confirm = input("Continue anyway? Y/N: ")
        if confirm.lower() not in ('y', 'yes'):
            print("A smaller division setting will reduce the number of tris.")
            return

    options = load_settings("options.json")
    fmt = options["mesh_format"]

    # meshio mesh
    # mesh = mi.Mesh(points, cells)
    print("Saving mesh to disk...")
    time_start = time.perf_counter()
    out_path = os.path.join(path, f"{name}.{fmt}")
    mi.write_points_cells(out_path, verts, {"triangle": tris})
    time_end = time.perf_counter()
    print(f"Mesh saved in {time_end - time_start :.5f} sec")

# ToDo: Implement this.  ply format, maybe obj? obj supports vertex data only?
# https://info.vercator.com/blog/what-are-the-most-common-3d-point-cloud-file-formats-and-how-to-solve-interoperability-issues
def save_point_cloud(verts, path, name):
    """Save the planet as a point cloud that can be used in other software."""

    options = load_settings("options.json")
    fmt = options["point_format"]

    print("Not implemented yet.")

def save_settings(data, path, name, fmt=None):
    """Save the variables that will recreate this planet."""
    out_path = os.path.join(path, f"{name}.{fmt}")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

# ToDo: Handle error better if the file does not exist; probably should include a try/except as well.
def load_settings(path):
    """Load planet variables or Nixis options from a saved file."""
    if os.path.exists(path):
        with open(path, "rt") as f:
            o = f.read()
        options = json.loads(o)
    else:
        print("Path does not exist:", path)
        sys.exit(0)
    return options

# ToDo: Implement this.
def save_log(path, name, fmt=None):
    """Save a verbose debug log to a text file."""
    print("Not implemented yet.")

# ToDo: Implement this. I'm thinking sqlite if there's a good python library.
# https://sqlite.org/whentouse.html
def export_planet(data, path, name):
    """Export the planet as a database file."""
    print("Not implemented yet.")

# https://math.stackexchange.com/questions/2110160/find-percentage-value-between-2-numbers
def find_percent_val(minval, maxval, percent):
    """Find the percentage value between a min and max number.
    minval -- The low number of the range.
    maxval -- The high number of the range.
    percent -- A number from 0.0 to 100.0 (inclusive)."""
    if not 0.0 < percent < 100.0:
        print("\n" + "    ERROR: Percent must be between 0 and 100 (inclusive).")
        print("    Defaulting to 50 percent." + "\n")
        percent = 50.0

    return minval + ((maxval - minval) * percent / 100.0)

# https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast/29799815#29799815
@njit(cache=True)
def find_first(item, vec):
    """Return the index of the first occurence of item in vec."""
    for i in range(len(vec)):  # Seems a smidge faster than enumerate
        if item == vec[i]:
            return i
    # Safety fallback (probably shouldn't ever happen)
    return -1

# ToDo: This is already pretty fast for k=2500 but maybe try prange later.
@njit(cache=True)
def build_adjacency(triangles):
    """Build an array of adjacencies for each mesh vertex."""
    # The 12 original vertices have connectivity of 5 instead of 6 so for the
    # 6th number we assign -1 as there's no such thing as a vert with ID of -1
    adj = np.full((int((len(triangles)+4)/2), 6), -1, dtype=np.int32)

    for t in triangles:  # for i in prange(len(triangles)):
        # v0 = t[0]
        # v1 = t[1]
        # v2 = t[2]
        # i0 = find_first(-1, adj[v0])
        # i1 = find_first(-1, adj[v1])
        # i2 = find_first(-1, adj[v2])
        # adj[v0][i0] = v1
        # adj[v1][i1] = v2
        # adj[v2][i2] = v0

        adj[t[0]][find_first(-1, adj[t[0]])] = t[1]  # adj[triangles[i][0]][find_first(-1, adj[triangles[i][0]])] = triangles[i][1]  # Ugly as sin but I think this would work for prange
        adj[t[1]][find_first(-1, adj[t[1]])] = t[2]  # adj[triangles[i][1]][find_first(-1, adj[triangles[i][1]])] = triangles[i][2]
        adj[t[2]][find_first(-1, adj[t[2]])] = t[0]  # adj[triangles[i][2]][find_first(-1, adj[triangles[i][2]])] = triangles[i][0]
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
# ToDo: Try to search for patterns in the vert order again. Maybe there might
# be something like the winding order for even indexes is one way and the
# order for odd indexes is the other way (probably not, but worth checking).
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


# Based on https://stackoverflow.com/a/59634071
def approx_size(x, name, flag="SI"):
    """Find the byte size of a numpy array or convert a number of bytes into
    human-relevant terms.
    x -- Takes a numpy array, or can be an int or float of the number of bytes.
    Output will automatically be in the nearest relevant range, e.g. KB MB etc.
    name -- Name of the thing being measured (for print statements).
    flag -- Output can use SI units (e.g. Megabytes) or binary (e.g Mebibytes).
    """
    # https://physics.nist.gov/cuu/Units/binary.html
    units = {1000: ['KB', 'MB', 'GB', 'TB'],
             1024: ['KiB', 'MiB', 'GiB', 'TiB']}
    if flag == "SI":
        mult = 1000
    elif flag == "BINARY":
        mult = 1024

    if isinstance(x, np.ndarray):
        print(f"* {name} array:")
        print(f"*  dtype: {x.dtype}")
        print(f"*  {x.itemsize} bytes per item")
        print(f"*  items: {x.size:,}")
        ssize = sys.getsizeof(x)
        size = x.nbytes

        for num, unit in enumerate(units[mult]):
            ssize = ssize / mult
            size = size / mult
            if size < mult:
                print(f"*  sys size: ~{ssize} {unit}")
                print(f"*  npy size: ~{size:.5f} {unit}")
                break
            if num == len(units[mult]) - 1:
                print("*  npy size: Heckin' LARGE")

    elif isinstance(x, (int, float)):
        print(f"* {name}:")
        size = x

        for num, unit in enumerate(units[mult]):
            size = size / mult
            if size < mult:
                print(f"*  size: ~{size:.5f} {unit}")
                break
            if num == len(units[mult]) - 1:
                print("*  size: Heckin' LARGE")

    else:
        print("'The heck is this?")
        print(f"*  type: {type(x)}")


# https://stackoverflow.com/a/30316760
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
reject = type, ModuleType, FunctionType

def getsize(obj):
    """Sum size of object & members in bytes."""
    if isinstance(obj, reject):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for o in objects:
            print(o)
            if not isinstance(o, reject) and id(o) not in seen_ids:
                seen_ids.add(id(o))
                size += sys.getsizeof(o)  # sys.getsizeof fails to get the proper size for np float64 arrays but works with int32 and float32 arrays because reasons? Or it's measuring the wrong reference maybe.
                need_referents.append(o)
        objects = get_referents(*need_referents)
    return size  # Returns size of obj in bytes

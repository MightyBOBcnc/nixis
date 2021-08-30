"""Module of erosion functions."""
import random
import numpy as np
from numba import jit, njit, prange
from util import *
# pylint: disable=not-an-iterable

# ToDo:
# Split erosion into functions by type.
#   A function for hydraulic/fluvial erosion.  https://en.wikipedia.org/wiki/Fluvial_processes
#     Shallow Water Simulation is another name to look up for papers. (A simplification of Navier Stokes equations?)
#   A function for thermal erosion (gravity/slope erosion) (NOTE: Deserts have stronger thermal weathering due to the extreme day/night temperature swings expanding/contracting stone.)
#   A function for INVERSE thermal? As described in "Procedurally Generating Terrain - mics2011_submission_30.pdf"
#   A function for aeolian (wind) eroosion? (NOTE: The effects of such erosion are likely very small at the scale Nixis works at)
#   A function for glacial erosion
#   A function for ocean/sea/lake beach erosion
#   A function for ocean current (ocean floor) erosion (NOTE: The deeper a water body is, the less erosion there is at the floor. This is because water flow is extremely slow at high depths. Sediment capacity is enormous, however.)
# Function for making a graph of river flow
#   Use graph for svg output of rivers
#   Use graph for grouping/splitting/showing individual water sheds
# Track soil movement for nutrients and fertility (good places for plant growth)
# Salinity of water bodies (salt lakes, mainly)

@njit(cache=True)
def calc_slope(v0, v1, dist):
    return (v1 - v0) / (dist + 0.00001)

@njit(cache=True)
def calc_distance(v0, v1):
    return np.sqrt( (v0[0] - v1[0])**2 + (v0[1] - v1[1])**2 +(v0[2] - v1[2])**2 )

@njit(cache=True)
def erode_terrain1(nodes, neighbors, heights, num_iter=1, snapshot=None):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1

    # read_buffer = heights.view(dtype=np.float64)  # Numba no like for some reason

#    print("Input height object:", id(heights))

    read_buffer = heights
    write_buffer = np.ones_like(heights, dtype=np.float64)

#    print("Read buffer object: ", id(read_buffer))
#    print("Write buffer object:", id(write_buffer))

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        write_buffer = erosion_iteration1(neighbors, read_buffer, write_buffer)
        # Switch the read and write buffers
        if i < num_iter:                           # ToDo: This is a mess. Why does this line even exist?  Also read_buffer has no purpose.  Heights is the read buffer.  We only need the write_buffer to hold temp values.
            for x in prange(len(write_buffer)):    # Also we should be writing straight into heights. The layout of the function should be write_buffer = erosion_iteration to store the values for the iteration,
                read_buffer[x] = write_buffer[x]   # and then heights[x] = write_buffer[x].  There doesn't even need to be a return value for the function.
        # read_buffer = write_buffer
        # write_buffer = read_buffer
#        print("Read buffer object: ", id(read_buffer))
#        print("Write buffer object:", id(write_buffer))

    if len(neighbors) < 43:
        print("New heights:")
        print(write_buffer)
    # Return new height values.. or return heights = write_buffer (handle height replacement in this func instead of outside because we won't be needing the unmodified heights anymore)
    return read_buffer

@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration1(neighbors, r_buff, w_buff):
    simple_constant = 0.0005
    # For every vertex index
    for i in prange(len(neighbors)):
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Agregate amount by which to change height
        amt = 0
        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                # compvert = r_buff[n]  # ToDo: Is it more performant to assign r_buff[n] to a var or to read it twice in the below comparison?
                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                # Do math to determine if compvert is higher or lower than thisvert
                if r_buff[n] > thisvert:
                    amt += simple_constant
                elif r_buff[n] < thisvert:
                    amt -= simple_constant

        w_buff[i] = thisvert + amt
        # print("after:", w_buff[i])

    return w_buff

# =========================

@njit(cache=True)
def erode_terrain2(nodes, neighbors, heights, num_iter=1, snapshot=None):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1

    water = np.zeros_like(heights)  # ToDo: Dtypes!  Save RAM?!
    sediment = np.zeros_like(heights)  # ToDo: Dtypes!  Save RAM?!

#    print("Input height object:", id(heights))


    write_buffer = np.ones_like(heights, dtype=np.float64)

#    print("Read buffer object: ", id(read_buffer))
#    print("Write buffer object:", id(write_buffer))

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        erosion_iteration2(nodes, neighbors, heights, water, sediment, write_buffer)
        # Switch the read and write buffers
        for x in prange(len(write_buffer)):
            heights[x] = write_buffer[x]
        # read_buffer = write_buffer
        # write_buffer = read_buffer
#        print("Read buffer object: ", id(read_buffer))
#        print("Write buffer object:", id(write_buffer))

    if len(neighbors) < 43:
        print("New heights:")
        print(write_buffer)

# Idea inspired by Axel Paris' description of aggregating the influence of each neighbor, with a write buffer to avoid race conditions.
# https://perso.liris.cnrs.fr/aparis/public_html/posts/terrain_erosion.html
# https://perso.liris.cnrs.fr/aparis/public_html/posts/terrain_erosion_2.html

@njit(cache=True, parallel=True, nogil=True)                         # I think I accidentally implemented thermal erosion instead of hydraulic erosion because this doesn't use or transport the water or sediment yet.
def erosion_iteration2(verts, neighbors, r_buff, wat, sed, w_buff):  # However the erosion doesn't have a cutoff angle like talus slippage (talus slips only happen at steep slopes) so it's more akin to a gaussian blur.
    simple_constant = 0.05                                           # I'm also seeing hex patterns emerge after more iterations.

    # For every vertex index
    for i in prange(len(neighbors)):
        # random.seed(i)  # This produces an interesting terraced effect but is suuuper slow
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Agregate amount by which to change height
        amt = 0
        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                compvert = r_buff[n]
                # Note: d is using the UNMODIFIED vert positions from
                # the smooth icosphere before any height modification
                d = calc_distance(verts[i], verts[n])
                slope = calc_slope(thisvert, compvert, d)
                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                # Do math to determine if compvert is higher or lower than thisvert
                if slope > 0:  # Neighbor is higher than this vert
                    amt += simple_constant * d * random.random()  # Multiplying by a constant like 0.7 instead of random.random produces its own interesting result; as does not multiplying by anything at all (simple_constant * d only)
                elif slope < 0:  # Neighbor is lower than this vert
                    amt -= simple_constant * d * random.random()

        w_buff[i] = thisvert + amt
        # print("after:", w_buff[i])


# =========================

# @njit(cache=True)
def erode_terrain3(nodes, neighbors, heights, num_iter=1, snapshot=None):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1

    water = np.zeros_like(heights)  # ToDo: Dtypes!  Save RAM?!
    sediment = np.zeros_like(heights)  # ToDo: Dtypes!  Save RAM?!

    if snapshot is not None:
        my_dir = os.path.dirname(os.path.abspath(__file__))
        with open("options.json", "rt") as f:  # ToDo: Handle error if the file does not exist
            options = json.load(f)
        save_dir = os.path.join(my_dir, options["save_folder"], "snapshots")
        try:  # ToDo: Test if the directory already exists. Maybe even attempt to see if we have write permission beforehand.
            os.mkdir(save_dir)
        except:
            # ToDo: Actual exception types
            # print("Failed to create script output directory!")
            pass

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        rain_amount = 0.3 / 320# * random.random()
        water += rain_amount
        erosion_iteration3(nodes, neighbors, heights, water, sediment)

        if snapshot is not None:  #ToDo: This is quite slow.  Also it's a dumb hack.
            dictionary = {}
            rescaled_h = rescale(heights, 0, 255)  #NOTE: Due to the relative nature of rescale, if the min or max height changes then the scale will be messed up.
            dictionary[f"{i+1:03d}"] = rescaled_h

            pixel_data = build_image_data(snapshot, dictionary)
            save_image(pixel_data, save_dir, "erosion_snapshot")

# Attempting Travis Archer's description of hydraulic erosion combined with the race condition fix described by Axel Paris.
# Procedurally Generating Terrain - mics2011_submission_30.pdf

@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration3(verts, neighbors, r_buff, wat, sed):
    height_buffer = np.copy(r_buff)
    water_buffer = np.copy(wat)
    sed_buffer = np.copy(sed)

    simple_constant = 0.05

    evaporation = 0.1 / 320# * random.random()
    solubility = 0.01 / 320
    capacity = 0.2 / 320

    # For every vertex index
    for i in prange(len(neighbors)):
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Agregate amount by which to change height
        sed_amt = sed[i]
        wat_amt = wat[i]

        # print(" === Vertex:   ", i)
        # print("Start height:  ", thisvert)
        # print("Start sediment:", sed_amt)
        # print("Start water:   ", wat_amt)

        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                compvert = r_buff[n]
                # Do math to determine if compvert is higher or lower than thisvert
                # Note: d is using the UNMODIFIED vert positions from
                # the smooth icosphere before any height modification
                d = calc_distance(verts[i], verts[n])
                slope = calc_slope(thisvert, compvert, d)

                # print("  Neighbor vert:", n)
                # print("  Nbr height:   ", compvert)
                # print("  Nbr distance: ", d)
                # print("  Nbr slope:    ", slope)

                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                if slope > 0:  # Neighbor is higher than this vert
                    sed_amt += solubility * wat[n]# * d
                    wat_amt += wat[n] * d

                    # print(" Adding", max(solubility * wat[n], 0), "to sed_amt")
                    # print(" Adding", max(wat[n] * d, 0), "to wat_amt")

                elif slope < 0:  # Neighbor is lower than this vert
                    sed_amt -= solubility * wat[n]# * d
                    wat_amt -= wat[n] * d

                    # print(" Subtracting", max(solubility * wat[n], 0), "from sed_amt")
                    # print(" Subtracting", max(wat[n] * d, 0), "from wat_amt")

                else:
                    print("  Doing Nothing.")

# Technically speaking we are not handling cases where the slope is == 0 because the height is the same.
# It's also possible that for cases with very low slope (nearly the same height) we are adding or subtracting TOO much to sediment and water.  And water doesn't use a pressure model.
# It should also be noted that this algorithm doesn't really have a concept of cell/area.

        # print(" sed_amt:", sed_amt)
        # print(" wat_amt:", wat_amt)

        height_buffer[i] -= sed_amt
        sed_buffer[i] += sed_amt
        water_buffer[i] += wat_amt - wat_amt * evaporation
        if sed_buffer[i] > (capacity * water_buffer[i]):
            height_buffer[i] += (sed_buffer[i] - (capacity * water_buffer[i]))
            sed_buffer[i] -= (sed_buffer[i] - (capacity * water_buffer[i]))

        # print("End height:  ", height_buffer[i])
        # print("End sediment:", sed_buffer[i])
        # print("End water:   ", water_buffer[i])

    # Switch the read and write buffers
    for x in prange(len(r_buff)):
        r_buff[x] = height_buffer[x]
        sed[x] = sed_buffer[x]
        wat[x] = water_buffer[x]
        # sed[x] = 0  # Resetting sediment doesn't help
        # wat[x] = 0  # Resetting water to 0 doesn't help; and doing both at the same time negates the simulation...

# =========================

@njit(cache=True)
def erode_terrain4(nodes, neighbors, heights, num_iter=1, snapshot=None):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1

    water = np.zeros_like(heights)  # ToDo: Dtypes!  Save RAM?!
    sediment = np.zeros_like(heights)  # ToDo: Dtypes!  Save RAM?!

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        rain_amount = 0.3 / 320# * random.random()
        water += rain_amount
        erosion_iteration3(nodes, neighbors, heights, water, sediment)

# Attempting Travis Archer's description of hydraulic erosion combined with the race condition fix described by Axel Paris.
# Procedurally Generating Terrain - mics2011_submission_30.pdf

@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration4(verts, neighbors, r_buff, wat, sed):
    height_buffer = np.copy(r_buff)
    water_buffer = np.copy(wat)
    sed_buffer = np.copy(sed)

    simple_constant = 0.05

    evaporation = 0.1 / 320# * random.random()
    solubility = 0.01 / 320
    capacity = 0.2 / 320

    # For every vertex index
    for i in prange(len(neighbors)):
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Agregate amount by which to change height
        sed_amt = sed[i]
        wat_amt = wat[i]

        # print(" === Vertex:   ", i)
        # print("Start height:  ", thisvert)
        # print("Start sediment:", sed_amt)
        # print("Start water:   ", wat_amt)

        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                compvert = r_buff[n]
                # Do math to determine if compvert is higher or lower than thisvert
                # Note: d is using the UNMODIFIED vert positions from
                # the smooth icosphere before any height modification
                d = calc_distance(verts[i], verts[n])
                slope = calc_slope(thisvert, compvert, d)

                # print("  Neighbor vert:", n)
                # print("  Nbr height:   ", compvert)
                # print("  Nbr distance: ", d)
                # print("  Nbr slope:    ", slope)

                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                if slope > 0:  # Neighbor is higher than this vert
                    sed_amt += max(solubility * wat[n], 0)# * d
                    wat_amt += max(wat[n] * d, 0)

                    # print(" Adding", max(solubility * wat[n], 0), "to sed_amt")
                    # print(" Adding", max(wat[n] * d, 0), "to wat_amt")

                elif slope < 0:  # Neighbor is lower than this vert
                    sed_amt -= max(solubility * wat[n], 0)# * d
                    wat_amt -= max(wat[n] * d, 0)

                    # print(" Subtracting", max(solubility * wat[n], 0), "from sed_amt")
                    # print(" Subtracting", max(wat[n] * d, 0), "from wat_amt")

                else:
                    print("  Doing Nothing.")

# Technically speaking we are not handling cases where the slope is == 0 because the height is the same.
# It's also possible that for cases with very low slope (nearly the same height) we are adding or subtracting TOO much to sediment and water.  And water doesn't use a pressure model.
# It should also be noted that this algorithm doesn't really have a concept of cell/area.

        # print(" sed_amt:", sed_amt)
        # print(" wat_amt:", wat_amt)

        height_buffer[i] -= max(sed_amt, 0)
        sed_buffer[i] += max(sed_amt, 0)
        water_buffer[i] += max((wat_amt - wat_amt * evaporation), 0)
        if sed_buffer[i] > (capacity * water_buffer[i]):
            height_buffer[i] += max((sed_buffer[i] - (capacity * water_buffer[i])), 0)
            sed_buffer[i] -= max((sed_buffer[i] - (capacity * water_buffer[i])), 0)

        # print("End height:  ", height_buffer[i])
        # print("End sediment:", sed_buffer[i])
        # print("End water:   ", water_buffer[i])

    # Switch the read and write buffers
    for x in prange(len(r_buff)):
        r_buff[x] = height_buffer[x]
        sed[x] = sed_buffer[x]
        wat[x] = water_buffer[x]

# =========================

#@njit(cache=True)
def erode_terrain5(nodes, neighbors, heights, num_iter=1, snapshot=None):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        erosion_iteration5(nodes, neighbors, heights, i)

# Attempting Jason Rampe's method.
# https://softologyblog.wordpress.com/2016/12/09/eroding-fractal-terrains-with-virtual-raindrops/

@njit(cache=True)
def find_lowest(vert, nbr, height):
    alts = [height[i] for i in nbr if i != -1]
    lowest = min(alts)

    if lowest < height[vert]:
        return find_first(lowest, alts)
    else:
        return None

#@njit(cache=True, parallel=False, nogil=False)
def erosion_iteration5(verts, neighbors, r_buff, iteration):
    height_buffer = np.copy(r_buff)

    num_drops = len(verts)# * 2
    erode_rate = 0.0001
    deposit_rate = 0.0001

    np.random.seed(iteration)
    drop_starts = np.random.randint(0, len(verts), size=num_drops)
    carried_soil = 0

    for i in range(num_drops):
        drop_loc = drop_starts[i]
        dest = find_lowest(drop_loc, neighbors[drop_loc], r_buff)
        # print("Start loc:", drop_loc)
        # print("Next loc:", dest)

        while dest is not None:
            d = calc_distance(verts[drop_loc], verts[dest])
            slope = calc_slope(r_buff[drop_loc], r_buff[dest], d)

            pick_up = slope * erode_rate
            carried_soil += pick_up

            height_buffer[drop_loc] -= max(pick_up, 0)
            depo = pick_up * deposit_rate * slope
            height_buffer[drop_loc] += depo
            carried_soil -= max(depo, 0)

            prev_loc = drop_loc
            drop_loc = dest
            dest = find_lowest(drop_loc, neighbors[drop_loc], r_buff)
            if dest != prev_loc:
                continue
            else:
                dest = None
            # print("Next loc:", dest)
        else:
            height_buffer[drop_loc] += carried_soil


    # Switch the read and write buffers
    for x in prange(len(r_buff)):
        r_buff[x] = height_buffer[x]

# =========================

def erode_terrain6(cells):
    print("Not implemented yet")

    # for c in cells:
    #     for i, j in neighbor_pairs:
    #         aggregates[i] += elevation[j]
    #         aggregates[j] += elevation[i]
# In the end, you divide by 2 because you counted each contribution twice. Of course, you wouldn't use explicit Python loops, but np.add.at or npx.sum_at.
# https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
# https://github.com/nschloe/npx#sum_atadd_at


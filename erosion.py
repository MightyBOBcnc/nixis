"""Module of erosion functions."""
import random
import numpy as np
from numba import njit, prange
import cfg
from util import *
# pylint: disable=not-an-iterable
# pylint: disable=line-too-long

# TODO:
# Split erosion into functions by type.
#   A function for hydraulic/fluvial erosion.  https://en.wikipedia.org/wiki/Fluvial_processes
#     Shallow Water Simulation is another name to look up for papers. (A simplification of Navier Stokes equations?)
#   A function for thermal erosion (gravity/slope erosion) (NOTE: Deserts have stronger thermal weathering due to the extreme day/night temperature swings expanding/contracting stone.)
#     NOTE: Thermal erosion is probably irrelevant at the scales Nixis is working at.
#   A function for INVERSE thermal? As described in "Procedurally Generating Terrain - mics2011_submission_30.pdf"
#   A function for aeolian (wind) eroosion? (NOTE: The effects of such erosion are likely very small at the scale Nixis works at)
#   A function for glacial erosion
#   A function for ocean/sea/lake beach erosion
#     https://en.wikipedia.org/wiki/Coast#Geologic_processes
#     https://en.wikipedia.org/wiki/Beach#Erosion_and_accretion
#   A function for ocean current (ocean floor) erosion
#     (NOTE: The deeper a water body is, the less erosion there is at the floor. This is because water flow is extremely slow at high depths. Sediment capacity is enormous, however.)
#     https://en.wikipedia.org/wiki/Turbidite
# Function for making a graph of river flow
#   Use graph for svg output of rivers
#   Use graph for grouping/splitting/showing individual water sheds
# Track soil movement for nutrients and fertility (good places for plant growth)
# NOTE: The USGS Landsat dataset tracking crop growth is beautiful. You can see the exact locations where all the fields for each crop type are, which correlates with geography and soil fertility, of course. It's a data set you rarely see.
#       Mostly one just thinks of crops generically at the state level. It's amazing to see the small localized areas for certain crops and realize that we're all depending on this small area for that crop (not counting imports, of course).
# Salinity of water bodies (salt lakes, mainly)
# This page is unrelated to erosion but I do want to try the 'virtual pipe' method, and this page mentions that the flow rate through a pipe is the square of the diameter? But it also says that if you increase the pressure it's the square root of pressure?
# http://homework.uoregon.edu/pub/class/es202/GRL/dwh.html
# Track ground water movement?  I like that leather bee was attempting this.
#   Useful info video, and mentions a resource for already computed values for different soil materials:
#      https://www.youtube.com/watch?v=bG19b06NG_w
#      Also this one has an equation: https://youtu.be/bY1E2IkvQ3k


# https://www.archtoolbox.com/representation/geometry/slope.html
@njit(cache=True)
def calc_slope(h0, h1, dist):
    """Calculates slope as a percentage. (returns percent as a decimal)

    h0 -- The height of the first vertex.
    h1 -- The height of the second vertex.
    dist -- The distance between the two vertices.
    """
    return (h1 - h0) / (dist + 0.00001)
# NOTE: Do we want our percentage as a decimal or full percentage?  We need to multiply by 100 to get a full percentage.  Currently returns as a decimal.
# TODO: Instead of adding a tiny amount to dist to prevent div/0, maybe do a conditional check instead
# and return a slope of 0 if there's no distance (or tiny distance) between the verts?
# In fact, could this be one of the reasons why the sim seems unstable and produces ridiculous spikes?
# i.e. dividing by a tiny decimal can make a large number (like, 5/0.0001 = 500,000)

@njit(cache=True)
def calc_slope_deg(h0, h1, dist):
    """Calculates slope as degrees.

    h0 -- The height of the first vertex.
    h1 -- The height of the second vertex.
    dist -- The distance between the two vertices.
    """
    return np.rad2deg(np.arctan( (h1 - h0) / (dist + 0.00001) ))

@njit(cache=True)
def calc_distance(v0, v1):
    """Calculates distance between two XYZ coordinates, in meters.

    v0 -- The XYZ coordinates of the first vertex.
    v1 -- The XYZ coordinates of the second vertex.
    v0 and v1 must both be an iterable with 3 items each.
    """
    # sqrt( (X2 - X1)^2 + (Y2 - Y1)^2 + (Z2 - Z1)^2 )
    return np.sqrt( (v1[0] - v0[0])**2 + (v1[1] - v0[1])**2 + (v1[2] - v0[2])**2 )

@njit(cache=True, parallel=False, nogil=False)
def check_world_slopes(heights, verts, nbrs):
    """Find the max and min slope on the planet. (For debugging.)"""
    slopes = np.zeros_like(nbrs, dtype=np.float64)
    for i in prange(len(nbrs)):
        for n in nbrs[i]:
            if n != -1:
                d = calc_distance(verts[i], verts[n])
                slope = calc_slope_deg(heights[i], heights[n], d)
                slopes[i][find_first(0, nbrs[i])] = slope
    print(" np.amax slope:", np.amax(slopes))
    print(" np.amin slope:", np.amin(slopes))
    slopes = None

# Simple erosion test. Add or subtract a constant based on height of neighbor verts.
# @njit(cache=True)
def erode_terrain1(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    # TODO: Should probably raise an exception if num_iter is invalid.
    # Yes we could 'gracefully' continue with a value of 1 but it's probably holistically better to stop.
    if num_iter <= 0:
        num_iter = 1
        print(" ERROR: Cannot have less than 1 iteration.")

    write_buffer = np.ones_like(heights, dtype=np.float64)

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        erosion_iteration1(neighbors, heights, write_buffer)
        # Switch the read and write buffers
        heights, write_buffer = write_buffer, heights

        if snapshot:
            save_snapshot(heights, i)

    # For odd number of iterations, ensure buffers get swapped back at the end
    # NOTE: Does this mean snapshots actually lag behind by 1 iteration?
    if num_iter % 2 != 0:
        heights, write_buffer = write_buffer, heights

    if len(neighbors) < 43:
        print("New heights:")
        print(heights)


@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration1(neighbors, r_buff, w_buff):
    simple_constant = 0.5
    # For every vertex index
    for i in prange(len(neighbors)):
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Aggregate amount by which to change height
        amt = 0
        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                # compvert = r_buff[n]  # TODO: Is it more performant to assign r_buff[n] to a var or to read it twice in the below comparison?
                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                # Do math to determine if compvert is higher or lower than thisvert
                if r_buff[n] > thisvert:
                    amt += simple_constant
                elif r_buff[n] < thisvert:
                    amt -= simple_constant

        w_buff[i] = thisvert + amt
        # print("after:", w_buff[i])


# =========================

# Idea inspired by Axel Paris' description of aggregating the influence of each neighbor, with a write buffer to avoid race conditions.
# https://perso.liris.cnrs.fr/aparis/public_html/posts/terrain_erosion.html
# https://perso.liris.cnrs.fr/aparis/public_html/posts/terrain_erosion_2.html

# Simple erosion test. Add or subtract a constant based on slope to neighbor verts, modified by distance.
# @njit(cache=True)
def erode_terrain2(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1
        print(" ERROR: Cannot have less than 1 iteration.")

    write_buffer = np.ones_like(heights, dtype=np.float64)

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        erosion_iteration2(nodes, neighbors, heights, write_buffer)
        # Switch the read and write buffers
        heights, write_buffer = write_buffer, heights

        if snapshot:
            save_snapshot(heights, i)

    # For odd number of iterations, ensure buffers get swapped back at the end
    # NOTE: Does this mean snapshots actually lag behind by 1 iteration?
    if num_iter % 2 != 0:
        heights, write_buffer = write_buffer, heights

    if len(nodes) < 43:
        print("New heights:")
        print(heights)


@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration2(verts, neighbors, r_buff, w_buff):  # I'm seeing hex patterns emerge after more iterations. It's also kinda noisy. The terraces are cool, though. Maybe combine with average_terrain function.
    # simple_constant = 0.05
    simple_constant = 0.0005
    # A constant of 0.05 works fine when the world_radius is 1.0 and the elevation spread is -0.05 to +0.15
    # and a constant of 0.0005 works with a world_radius of 6378100.0 and an elevation spread of -4000 to +8850
    # so what is the relation?  It's not linear because the orders of magnitude of difference between 1 and 6 million (6) is different than the order of magnitude of difference between 0.05 and 0.0005 (2).
    # Down below we are multiplying the constant times the distance between neighbor verts so it's related to the ratio of distance between verts and the world radius (which I've previously found is hard to calculate).
    # So maybe go back to that spreadsheet of distances between verts and see what those are when multiplied by 0.05 and 0.0005

    # For every vertex index
    for i in prange(len(verts)):
        # random.seed(i)  # This produces an interesting terraced effect but is suuuper slow
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Aggregate amount by which to change height
        amt = 0
        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                compvert = r_buff[n]
                # NOTE: d is using the UNMODIFIED vert positions from
                # the smooth icosphere before any height modification
                d = calc_distance(verts[i], verts[n])
                slope = calc_slope_deg(thisvert, compvert, d)
                # if i < -5:
                #     print(" Distance:", d)
                #     print(" Slope:   ", slope)
                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen

                # TODO: Problem: The radius of the planet, and the number of subdivisions are dynamic, which means that the distance between vertices and the slope are dynamic.
                # At the default radius (Earth) and subdivisions (320) the distance between vertices ranges from around 17000 to 26000 meters, which means the slope (in degrees)
                # is only ever like +/- 0.18 degrees at the most extreme. Increasing the divisions to 5000 brings the distance between verts down to "only" 1100 meters at the low end
                # but that leaves the slope as still a very small number. When the slope is very small that means there is effectively no thermal erosion because we'll never exceed the
                # talus angle if we set it to anything other than 0 (as we're currently doing below). At the moment it's set to 0 which is functional but erodes the entire terrain.
                # Thermal erosion should only erode above a certain angle so this isn't very realistic at the moment. Really, thermal erosion is a small-scale effect.
                # Bringing the radius of the planet down to only a few thousand meters does have the effect of shortening the distance between verts and increasing the slope angles.
                # For the default radius and divisions testing against a slope of +/- 0.1 or 0.2 or 0.3 below is useful for experimenting with different results.

                # Do math to determine if compvert is higher or lower than thisvert
                if slope > 0:  # Neighbor is higher than this vert.
                    amt += simple_constant * d * random.random()  # Multiplying by a constant like 0.7 instead of random.random produces its own interesting result; as does not multiplying by anything at all (simple_constant * d only)
                elif slope < 0:  # Neighbor is lower than this vert
                    amt -= simple_constant * d * random.random()

        w_buff[i] = thisvert + amt
        # print("after:", w_buff[i])


# =========================

# Attempting Travis Archer's description of hydraulic erosion combined with the race condition fix described by Axel Paris.
# Procedurally Generating Terrain - mics2011_submission_30.pdf

# More complex erosion attempt using water with a carrying capacity, sediment, and evaporation.
# NOTE: This is INCREDIBLY broken.
# NOTE: Should really be validating water transport before adding evaporation, before adding sediment.
# @njit(cache=True)
def erode_terrain3(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1
        print(" ERROR: Cannot have less than 1 iteration.")

    water = np.zeros_like(heights)  # TODO: Dtypes!  Save RAM?!
    sediment = np.zeros_like(heights)  # TODO: Dtypes!  Save RAM?!

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        rain_amount = 0.15# * random.random()  # Would like a psuedo-random amount of rain per loop.  0.15 meters is 6 inches.
        water += rain_amount  # Would also like to distribute it according to climate but for now all verts get it equally
        erosion_iteration3(nodes, neighbors, heights, water, sediment)

        if snapshot:
            save_snapshot(heights, i)


@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration3(verts, neighbors, r_buff, wat, sed):
    height_buffer = np.copy(r_buff)
    water_buffer = np.copy(wat)  # These and the parameters for the function really need better names.
    sed_buffer = np.copy(sed)  # Like explicitly name read and write like sed_read, sed_write.

    evaporation = 0.2# * random.random()
    solubility = 0.02
    capacity = 0.2

    testvert = -1  # Set me to a valid vertex index to gain insight into what is happening with the math

    # For every vertex index
    for i in prange(len(verts)):
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Aggregate amount by which to change height
        sed_amt = sed[i]
        wat_amt = wat[i]

        if i == testvert:
            # print("     Vertex:   ", i)
            print(" Start height:  ", thisvert)
            print(" Start sediment:", sed_amt)  # This is always 0.0?  Seems anomalous.
            print(" Start water:   ", wat_amt)

        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                compvert = r_buff[n]
                # Do math to determine if compvert is higher or lower than thisvert
                # NOTE: d is using the UNMODIFIED vert positions from
                # the smooth icosphere before any height modification
                d = calc_distance(verts[i], verts[n])
                delta = compvert - thisvert  # TODO: Reconcile that the slope functions subtract h1-h0 to get the rise but delta does the reverse here?
                slope = calc_slope(thisvert, compvert, d)
                # slope = calc_slope_deg(thisvert, compvert, d)

                # print("  Neighbor vert:", n)
                # print("  Nbr height:   ", compvert)
                # print("  Nbr distance: ", d)
                # print("  Nbr slope:    ", slope)

                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                # No inbuilt protection from exceeding the available water and sediment?  See erosion_iteration4
                if slope > 0:  # Neighbor is higher than this vert
                    sed_amt += solubility * wat[n]
                    wat_amt += wat[n]

                elif slope < 0:  # Neighbor is lower than this vert
                    sed_amt -= solubility * wat[n]
                    wat_amt -= wat[n]

                # else:
                #     print("  Doing Nothing.")

                if i == testvert:
                    # print(f"Distance between", i, "and", n, "is", d)  # Don't know why f-strings are giving me the float64 element type instead of its actual value..
                    print(f" Delta between", i, "and", n, "is", delta)
                    print(f" Slope between", i, "and", n, "is", slope)
                    print(f" Vert {i} sed_amt:", sed_amt)
                    print(f" Vert {i} wat_amt:", wat_amt)

# Technically speaking we are not handling cases where the slope is == 0 because the height is the same.
# It's also possible that for cases with very low slope (nearly the same height) we are adding or subtracting TOO much to sediment and water.  And water doesn't use a pressure model.
# It should also be noted that this algorithm doesn't really have a concept of cell/area.

        # print(" sed_amt:", sed_amt)
        # print(" wat_amt:", wat_amt)

        # HACK attempt to not exceed the delta between i and its neighbors.
        # Should this actually be 12?  Delta/2 for each neighbor? (the correct place to do this is inside for n in neighbors[i]:)
        sed_amt /= 6
        wat_amt /= 6

        height_buffer[i] -= sed_amt
        sed_buffer[i] += sed_amt
        water_buffer[i] += wat_amt - (wat_amt * evaporation)
        if i == testvert:
            print(" Adding", max(wat_amt - wat_amt * evaporation, 0), "to water_buffer")
            print(f" water_buffer[{i}] is now", water_buffer[i])
        if sed_buffer[i] > (capacity * water_buffer[i]):
            height_buffer[i] += sed_buffer[i] - (capacity * water_buffer[i])
            sed_buffer[i] -= sed_buffer[i] - (capacity * water_buffer[i])

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

# Same style as erode_terrain3, but trying (and failing) to stop spikes with min and max caps and some other changes.
# NOTE: A brute force way to partially limit spikes from growing out of control would be like: 
# for index, if index is the highest of all 5/6 neighbors, adding to this index is forbidden.
# That way peaks could not get taller but you could still deposit soil onto other verts.  Of course if you're depositing
# a huge amount onto a vert that isn't a peak it could still become a new huge spike.
# NOTE: This is INCREDIBLY broken.
@njit(cache=True)
def erode_terrain4(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1
        print(" ERROR: Cannot have less than 1 iteration.")

    water = np.zeros_like(heights)  # TODO: Dtypes!  Save RAM?!
    sediment = np.zeros_like(heights)  # TODO: Dtypes!  Save RAM?!

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        rain_amount = 0.15# * random.random()  # Would like a psuedo-random amount of rain per loop.  0.15 meters is 6 inches.
        water += rain_amount  # Would also like to distribute it according to climate but for now all verts get it equally
        erosion_iteration4(nodes, neighbors, heights, water, sediment)


@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration4(verts, neighbors, r_buff, wat, sed):
    height_buffer = np.copy(r_buff)
    water_buffer = np.copy(wat)
    sed_buffer = np.copy(sed)

    simple_constant = 0.05

    evaporation = 0.1 / 320# * random.random()
    solubility = 0.0001 / 320
    capacity = 0.0002 / 320

    testvert = -1  # Set me to a valid vertex index to gain insight into what is happening with the math

    # For every vertex index
    for i in prange(len(verts)):
        thisvert = r_buff[i]
        # print("Before:", thisvert)
        # Aggregate amount by which to change height
        sed_amt = sed[i]
        wat_amt = wat[i]

        if i == testvert:
            # print("     Vertex:   ", i)
            print(" Start height:  ", thisvert)
            print(" Start sediment:", sed_amt)  # This is always 0.0?  Seems anomalous.
            print(" Start water:   ", wat_amt)

        # Read neighbors
        for n in neighbors[i]:
            if n != -1:
                compvert = r_buff[n]
                # Do math to determine if compvert is higher or lower than thisvert
                # NOTE: d is using the UNMODIFIED vert positions from
                # the smooth icosphere before any height modification
                d = calc_distance(verts[i], verts[n])
                delta = compvert - thisvert  # TODO: Reconcile that the slope functions subtract h1-h0 to get the rise but delta does the reverse here?
                slope = calc_slope(thisvert, compvert, d)  
        # NOTE: Perhaps instead, utilize the 'willgive', 'willreceive' method so slope is always positive,
        # and then allocate as needed, while multiplying with the slope instead of the distance below.
        # or keep the negative and positive slopes for identifying slope direction and just multiply by abs(slope) instead of distance.
        # Allowing negative numbers definitely feels problematic, e.g. if there is more sediment or water flowing out of a vertex than flowing into it.
        # Adding/subtracting/multiplying negative numbers is almost certainly screwing with the results. (e.g. evaporation multiplying by a negative number?)
        # using abs() and handling addition/subtraction differently is probably a better idea.

                # print("  Neighbor vert:", n)
                # print("  Nbr height:   ", compvert)
                # print("  Nbr distance: ", d)
                # print("  Nbr slope:    ", slope)

                # This is where any information about the layers of sediment/rock/etc and the hardness of the exposed layer will happen
                if slope > 0:  # Neighbor is higher than this vert
                    sed_amt += max(solubility * wat[n], 0)
                    wat_amt += max(wat[n] * slope, 0)

                    # print(" Adding", max(solubility * wat[n], 0), "to sed_amt")
                    # print(" Adding", max(wat[n] * slope, 0), "to wat_amt")

                elif slope < 0:  # Neighbor is lower than this vert
                    sed_amt -= max(solubility * wat[n], 0)
                    wat_amt -= max(wat[n] * slope, 0)

                    # print(" Subtracting", max(solubility * wat[n], 0), "from sed_amt")
                    # print(" Subtracting", max(wat[n] * slope, 0), "from wat_amt")

                # else:
                #     print("  Doing Nothing.")

                if i == testvert:
                    # print(f"Distance between", i, "and", n, "is", d)  # Don't know why f-strings are giving me the float64 element type instead of its actual value..
                    print(f" Delta between", i, "and", n, "is", delta)
                    print(f" Slope between", i, "and", n, "is", slope)
                    print(f" Vert {i} sed_amt:", sed_amt)
                    print(f" Vert {i} wat_amt:", wat_amt)

# Technically speaking we are not handling cases where the slope is == 0 because the height is the same.
# It's also possible that for cases with very low slope (nearly the same height) we are adding or subtracting TOO much to sediment and water.  And water doesn't use a pressure model.
# It should also be noted that this algorithm doesn't really have a concept of cell/area.

        # print(" sed_amt:", sed_amt)
        # print(" wat_amt:", wat_amt)

        # HACK attempt to not exceed the delta between i and its neighbors.
        # Should this actually be 12?  Delta/2 for each neighbor? (the correct place to do this is inside for n in neighbors[i]:)
        sed_amt /= 6
        wat_amt /= 6

        height_buffer[i] -= max(sed_amt, 0)
        sed_buffer[i] += max(sed_amt, 0)
        water_buffer[i] += max(wat_amt - (wat_amt * evaporation), 0)
        # water_buffer[i] += max(wat_amt - evaporation, 0)
        if i == testvert:
            print(" Adding", max(wat_amt - wat_amt * evaporation, 0), "to water_buffer")
            print(f" water_buffer[{i}] is now", water_buffer[i])
        if sed_buffer[i] > (capacity * water_buffer[i]):
            height_buffer[i] += max(sed_buffer[i] - (capacity * water_buffer[i]), 0)
            sed_buffer[i] -= max(sed_buffer[i] - (capacity * water_buffer[i]), 0)

        # print("End height:  ", height_buffer[i])
        # print("End sediment:", sed_buffer[i])
        # print("End water:   ", water_buffer[i])

    # Switch the read and write buffers
    for x in prange(len(r_buff)):
        r_buff[x] = height_buffer[x]
        sed[x] = sed_buffer[x]
        wat[x] = water_buffer[x]

# =========================

# There needs to be a cap on how much soil you can add to a vertex or subtract from a vertex.
# For example, if you have a vertex A with an elevation of 10 with a neighbor B whose elevation is 20
# then at MOST you could subtract 5 from B to give to A and then both verts have a new elevation of 15.
# Otherwise soil would start flowing UPHILL from B to A and A would become higher.
# So we need to calculate the height delta between each vertex.

# But it's a bit more complicated than that because each vert has 5-6 neighbors.
# For the uphill neighbors it should be easy enough to find out the maximum contribution that each can
# give to the current vert before it starts to grow higher than any uphill neighbor.
# For downhill neighbors, however, we cannot know what THEIR other uphill neighbors can contribute so at best
# we only know what the current vert could contribute in isolation...
# ...unless we add another step to the parent function erode_terrainX after we accumulate an intermediary array and use that in a second function within the parent.
# Maybe just ignore downhill verts and only care about uphill verts.
# Maybe just divide the amount going to downhill verts by 5 or 6 or something and accept that soil will do improper things.

# =========================

# Attempting Jason Rampe's method.
# https://softologyblog.wordpress.com/2016/12/09/eroding-fractal-terrains-with-virtual-raindrops/

def erode_terrain5(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1
        print(" ERROR: Cannot have less than 1 iteration.")

    write_buff = np.copy(heights)

    # if snapshot:
    #     print("Starting height min is", np.min(heights), "and starting height max is", np.max(heights))

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        # print("  Height object:", id(heights))
        # print("  Write object:", id(write_buff))
        erosion_iteration5(nodes, neighbors, heights, write_buff, i)

        if snapshot:
            # print("Height min is", np.min(heights), "and height max is", np.max(heights))
            save_snapshot(heights, i)


@njit(cache=True)
def find_lowest(vert, nbrs, height):
    """Return the index of the lowest neighbor vertex or -1 if no lower neighbor.

    vert -- The index of the current vertex.
    nbrs -- The list of this vertex's 5 or 6 neighbor vertex indices.
    height -- An array of heights indexed the same as the vertices.
    """
    # If vert is -1 it isn't valid and therefore can't have a lowest neighbor
    if vert == -1:
        return -1

    alts = [height[i] for i in nbrs if i != -1]
    lowest = min(alts)

    if lowest < height[vert]:
        return nbrs[find_first(lowest, alts)]
    return -1


@njit(cache=True)
def find_highest(vert, nbrs, height):
    """Return the index of the highest neighbor vertex or -1 if no higher neighbor.

    vert -- The index of the current vertex.
    nbrs -- The list of this vertex's 5 or 6 neighbor vertex indices.
    height -- An array of heights indexed the same as the vertices.
    """
    # If vert is -1 it isn't valid and therefore can't have a highest neighbor
    if vert == -1:
        return -1

    alts = [height[i] for i in nbrs if i != -1]
    highest = max(alts)

    if highest > height[vert]:
        return nbrs[find_first(highest, alts)]
    return -1


# This is super slow because the water drops run sequentially one at a time, not in parallel.
# Otherwise they would collide when changing heights (aka a race condition in the write_buffer).
# It does produce some interesting results, but they are highly subdivision-dependant.
@njit(cache=True, parallel=False, nogil=False)
def erosion_iteration5(verts, neighbors, height_read, write_buffer, iteration):
    num_drops = len(verts)# * 2
#    num_drops = 1
    erode_rate = 1.0001
    # Maybe deposit_rate actually should be <1 ?  We should be depositing less than we are eroding?
    # (yes, we intentionally lose soil mass by discarding carried_soil when the while loop terminates without depositing everything left in carried_soil)
    deposit_rate = 1.0001

    # Pseudo-random list of locations where we'll spawn drops.
    np.random.seed(iteration)
    drop_starts = np.random.randint(0, len(verts), size=num_drops)

    # NOTE: This loop can be run with prange but will make slightly different (aka non-deterministic)
    # outputs due to collisions when modifying the write_buffer.  Don't forget to set the
    # parallel=True parameter in the @njit decorator if you change this to prange
    for i in range(num_drops):
        drop_loc = drop_starts[i]
        # drop_loc = i
        dest = find_lowest(drop_loc, neighbors[drop_loc], height_read)
        # print("Start loc:", drop_loc)
        # print("Next loc:", dest)
        carried_soil = 0.0

        # TODO: I am not sure if this is a problem but it is worth noting (everything seems to work fine despite this observation):
        # When we find 'dest' and when we calc the 'slope' both above and inside the while loop we are using the read buffer (height_read).
        # However, the while loop makes changes to the write buffer and then *goes back to reading from the read buffer* on its next loop
        # but now the read buffer values are *outdated* because the water drop changed some things in the previous loop and these haven't been propagated
        # back into the read buffer.  So it's like it's following ghost terrain for the next drops that pass through any modified vertex.
        # Then eventually when the while loop terminates for having no lower destination we continue with the outer loop and start a new water drop but
        # we *still* are only doing reads from the read buffer so each drop that runs is running on an increasingly outdated set of information that
        # is more and more divergent from what's happening in the write buffer.
        # Only when we finally run out of drops do we actually copy all the changes from the write buffer back into the read buffer, and then the outer
        # function moves on to its next loop and calls this function again.
        #
        # Because this drop based function is different from all the other erosion functions (in that it only ever runs one drop at a time) should we
        # actually read from and write to the same array?  There can't be a race condition because nothing runs in parallel.
        # That would make for a problem if we try to get drops running in parallel in the future, but we'll have to change the structure then anyway so, meh.

        while dest != -1:
            d = calc_distance(verts[drop_loc], verts[dest])
            # TODO: Slope has essentially no meaning at large scales and is only meaningful for local erosion, not global erosion,
            # so its use here (or anywhere in Nixis) is technically wrong. Something to consider for further work.
            slope = calc_slope_deg(height_read[dest], height_read[drop_loc], d)
            # if i < 5:
            #     print(slope)

            # Slope on these global scales is usually <1, like 0.15, so this would be 0.15*1.0001 = 0.150015 picked up
            pick_up = slope * erode_rate
            # pick_up = height_read[drop_loc] / height_read[dest] * 0.5
            # pick_up = (height_read[drop_loc] - height_read[dest]) * 0.25
            carried_soil += max(pick_up, 0)  # Add picked up soil to carried_soil
            write_buffer[drop_loc] -= max(pick_up, 0)  # Remove same amount from current location

#            prev_loc = drop_loc
            drop_loc = dest  # Move water drop

            # Should really be checking the height delta between source and dest so we can't deposit more than half(?) that delta
            # Since the slope is usually <1 this means carried_soil (0.15) * deposit_rate (1.0001) divided by 1 / 0.15 (6.666) means .15 / 6.66 = deposit like 0.022
            # This might mean that if deposit_rate > 1 we actually deposit more than we started with?
            depo = max(carried_soil * deposit_rate / (1 / slope), 0)
            # depo = max(carried_soil * deposit_rate * slope, 0)
            write_buffer[drop_loc] += depo  # Deposit part of the carried soil at the new location
            carried_soil -= max(depo, 0)  # Remove same amount from carried soil

            dest = find_lowest(drop_loc, neighbors[drop_loc], height_read)
            # print("Next loc:", dest)
#            if dest != prev_loc:  # This if/else block shouldn't be needed because dest returns -1 if there's no lower neighbor
#                continue          # Upon second thought this might be needed to prevent bouncing back and forth at the final vert.. mayybe...
#            else:
#                dest = -1
        # else:  # this else shouldn't be needed; we just un-indent the below line by 1 and let it run when the while loop exits
#        write_buffer[drop_loc] += carried_soil  # Deposit any remaining carried soil.  Commented out because it might be depositing large amounts that are taller than the source they came from.


    # Only write to the original height array at the end (stop race conditions)
    # TODO: This is probably better: https://numpy.org/doc/stable/reference/generated/numpy.copyto.html
    # np.copyto(height_read, write_buffer, 'no')  # NEVERMIND, numba doesn't support np.copyto() yet (2023-08-25)
    for x in prange(len(height_read)):
        height_read[x] = write_buffer[x]


# =========================

# Supposed to be the same type of result as erode_terrain5 but threadsafe,however the output is more akin to
# ridged perlin noise accentuating the existing highs and lows instead of carving water channels. (Which is neat, but not the goal)
def erode_terrain5p(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    num_drops = len(nodes)
    write_buff = np.copy(heights)

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
#        rng = np.random.default_rng(i)  # Control the seed for each loop so it remains deterministic
        # TODO: Rebuilding drops array on every loop is probably bad.
        # We could actually build it once above this loop and pass it in, and then simply shuffle it for each iteration.
        # However this only works if we don't change the length of the array inside the iteration.
        # So basically we DO have to essentially track every vertex and know whether it has a drop on it.
        # It also won't work if we change any of its items...
#        drops = np.arange(num_drops, dtype=np.int32)  # One drop for every vertex
#        rng.shuffle(drops)  # Randomizes the order, but guarantees that each vertex gets a drop

        erosion_iteration5p(nodes, neighbors, heights, write_buff)

        if snapshot:
            save_snapshot(heights, i)


from numba.experimental import jitclass
from numba import int32, float32

@jitclass
class WaterDroplet():
    """Very simple water droplet object."""
    prev_loc: int32  # Drop's previous index
    cur_loc: int32  # Drop's current index
    next_loc: int32  # Drop's destination index

    def __init__(self, prev_loc, cur_loc, next_loc):
        self.prev_loc = prev_loc
        self.cur_loc = cur_loc
        self.next_loc= next_loc

    def debug(self):
        print("prev_loc:", self.prev_loc)
        print("cur_loc:", self.cur_loc)
        print("next_loc:", self.next_loc)

# TODO: Random thought right before bed.  Instead of the thing right below here, do a dict where the key is the vertex index, and the value is a WaterDrop class object. (can't do dest as key because dict can't have dupes)
# Then we just iterate over every vertex id (prange(len(verts))) and for each vertex, for n in its neighbors, check if dict[n] exists, and if it does access the carried soil and accumulate onto
# this vertex id either in a soil array or on another WaterDrop object for this id.. might need two dicts?  or the drop just knows the soil at its source and its dest and then we change them accordingly.
# Only problem is what to do if this vertex id already has a WaterDrop.. so yeah we might need two dicts or whatever.
# Or maybe some sort of dict where we just add to a list as the value for any dupes on that index (key).
# I don't know how numba would handle a dict with key/value pairs where the value can have a variable length, like a list that could contain 1 to 6 items, but...
# A way to do a dict for tracking water drops would be that the key is the vertex id, and the value is a list or array that always has a length of 6.  This is where any dupes would be stored.
# That way any for loop lookup would only ever have to search 6 entries (because you can't have more than 6 neighbors) instead of for each drop check EVERY other drop (horrible Big O time).
# Or, instead of this dict idea.. just make an array the same size and shape as the neighbors array (build_adjacency in util.py) and instead of storing the indices of a vertex's neighbors,
# use the 6 "slots" to hold water drop class objects, since there can never be more than 6 incoming drops from 6 higher neighbors. To deduplicate, just move the contents of drops 1 to 5
# onto drop 0 and destroy the drops in slots 1 to 5.  Of course, this all depends on whether numpy/numba will allow the slot to be either a class object or like, None or something.
# If not, then the dict idea is likely the only way to go, and I don't know if numba will like a dict with different-length (but always same type) values.

# 0. rename drop_locs to random_order (NOTE: think of the future when we have working climate that says where to rain; it won't actually be random so the name matters)
# 1. for i in vertices: dests[i] = find lowest(i, neighbors[i], height_read)  # dests indexed to vertices, not that weird shit above.
# 2.    if dests[i] != -1: calc distance and slope and set soil_read[i], and remove that amount from height_write[i]  # also indexed to vertices.
# 3. for i in random_order:
#       if the dests[random_order[i]] != -1:
#           for n in neighbors[random_order[i]] (or is it for n in neighbors of the dests[random_order[i]]?):
#               if random_order[i] is the destination of dests[n] or something like that:
# 4.                accumulate the soil_read[n] onto the soil_write[i] so that there isn't a race condition
#                   I think we can also deposit to height_write here, and subtract the same amount from soil_write, but we have to recalculate the distance and the slope between here and n :(
# 5. and I think here at the end we do a lot of this:
#     for x in prange(len(height_read)):
#         height_read[x] = height_write[x]
#    for the height and the soil (because we can't tuple swap the soil, it has to be cloned like the heights); and then I have to figure out what to do with dests... also all this goes in the previously mentioned while loop
@njit(cache=True, parallel=True, nogil=True)
def erosion_iteration5p(verts, neighbors, height_read, height_write):
#    num_drops = len(verts)
    erode_rate = 1.0001
    deposit_rate = 1.0001
    # potato = WaterDroplet(-1, -1, -1)
    # potato.debug()

    dests = np.full_like(height_read, -1, dtype=np.int32)
    soil_read = np.zeros_like(height_read, dtype=np.float32)  # TODO: This should be passed in
    soil_write = np.zeros_like(height_read, dtype=np.float32)

    # Set initial destinations for this iteration
    for i in prange(len(verts)):
        dests[i] = find_lowest(i, neighbors[i], height_read)

    # TODO: For some reason using "while True:" along with a test at the bottom of the while loop with a break condition never terminates
    # and I haven't sought to track down why this is the case, yet. For now the while loop is artificially restricted with a counter.
    # The height_read array that is the source for finding the downhill neighbors doesn't get modified until the end of the iteration
    # outside of the while loop so it SHOULD be possible for the loop to terminate with the np.amax check at the bottom...
    # NOTE: I think it's a logic flaw.  Down at the bottom of Part Two we find a new destination for EVERY vertex again instead of
    # restricting the search to exclude upstream verts that we previously came from.  Unless the entire world is flat this means
    # there will ALWAYS be new destinations because we effectively start the search over again by including the higher elevation verts.
    # We would need something like a set.difference() or check where if a vertex index DOESN'T appear in dests generated in Part One
    # then we exclude it from the new dests search at the bottom of Part Two.  e.g. if j not in dests, set dests[j] to -1.
    # The huge problem is that dests isn't a set.
    # I suppose we could make a set, and then in Part Two maybe inside the for n in neighbors we track if any of the 5/6 neighbors' dest
    # is me, and if not then add that vert to the exclude set.  This would require a little bit of extra code, perhaps an else statement
    # below 'if dests[n] == dest' that increments a counter and if the counter reaches 6 (or 5 for the first 12 verts) then add 'j' to
    # the exclude set.  Then down below we check if j is in exclude and set its dest to -1 when finding the new dests.
    # Q: Would we also call exclude.clear() to empty the set at the bottom of the while loop?
    # Or instead of wasting time with that set, we just set dests[j] to -1 when the counter reaches 5/6, and then down below
    # instead of checking if dests[j] is in the set, check if dests[j] != -1 (again) and only finding a new dest if that is true.
    # The one place where a set might be useful here is to just make a set of the neighbors right above 'for n in neighbors[dest]'
    # and see if j in set, then set -1 if no, and in fact we could totally skip the for n in neighbors part if that tests false.
    # The only question is whether instantiating a new set for every vertex on every loop like that would be more or less performant
    # than using a simple counter and for loop like we currently have (albeit commented out because it doesn't have the desired effect).
    c = 0
    while c < 8:  # NOTE: This can be set higher, but depending on the seed and divisions, spikes may form.
#    while True:
        # Part One
        for i in prange(len(verts)):
            if dests[i] != -1:
                d = calc_distance(verts[i], verts[dests[i]])
                slope = calc_slope_deg(height_read[dests[i]], height_read[i], d)
                pick_up = (slope * erode_rate)# * 0.5
                soil_read[i] += max(pick_up, 0.0)  # Pick up soil at current (source) vertex and set read array
                height_write[i] -= max(pick_up, 0.0)  # Remove same amount from current location

                # Can make for interesting ridges if the while loop doesn't run too many times
                # soil_read[i] += max(slope * erode_rate, 0)
                # height_write[i] -= max(soil_read[i], 0)

        # Part Two
        for j in prange(len(dests)):  # For the moment, don't even bother with random order, just loop through every vertex
            if dests[j] != -1:  # If current vertex index has a destination
                dest = dests[j]  # Index of the destination
#                exclusion_counter = 0
                for n in neighbors[dest]:  # n is index of dest's neighbor vertex
#                    if dests[n] == dest and n != -1:  # TODO: Problem. One of the neighbors, n, will be j, and dest will obviously be its destination, so the exclusion_counter may never reach 6?
                    if dests[n] == dest:  # If the destination of the neighbor is me
                        d = calc_distance(verts[j], verts[n])
                        slope = calc_slope_deg(height_read[j], height_read[n], d)

                    # Wait, how far indented does this go?  Do we want to modify the soil and height for each neighbor or at the end?
                    # the problem is that the 'depo' variable is modified by the slope, meaning if we wait to the end we need to know
                    # the slope of every neighbor.. and maybe average that or something.. But if we do it 1 at a time is the math correct?
                    # And more importantly if we do it 1 at a time will we be able to prange the for j in len(dests) or will that race condition?
                        # soil_write[j] += soil_read[n]

                # Depo could be defined as 0.0 right below dest, then add to it inside the for n in neighbors, increment a counter inside if dests[n] = dest, and finally divide by that counter to get an average of the slopes?
                # Or instead of that silliness just track slope the same way.. slope = 0.0, add to slope inside for n if dests[n] = dest, increment the counter and divide by it.
                        # if slope == 0.0:
                        #     depo = 0.0
                        # else:
                        depo = max(soil_read[n] * deposit_rate / (1.0 / slope), 0.0)# * 0.5  # * 0.16
#                        if dest == 8814131:
#                            print("Depositing", depo)
                        height_write[j] += depo
                        soil_write[j] -= max(depo, 0.0)  # TODO: This could possibly go into negative numbers
#                    else:
#                        exclusion_counter += 1
#                if exclusion_counter == 6:# or (j < 12 and exclusion_counter == 5):  # I am no one's destination
#                    dests[j] = -1


#            if dests[j] != -1:
#                if find_highest(j, neighbors[j], height_read) == -1:
#                    dests[j] = -1
#                else:
                dests[j] = find_lowest(j, neighbors[j], height_read)

        # for p in prange(len(dests)):
        #     dests[p] = find_lowest(p, neighbors[p], height_read)

        c += 1
        # print(np.amax(dests))
#        if np.amax(dests) == -1:
#            break

    for x in prange(len(verts)):
        height_read[x] = height_write[x]
        soil_read[x] = soil_write[x]
#        if x == 8814131:
#            print("height is", height_write[x])


    # To move the drop in the single thread we simply do cur_loc = dest
    # I think here we could actually do the same because cur_loc is already mapped to drop_locs[i] and same for dest and dests[i] ??
    # But then that breaks carried_soil[cur_loc] doesn't it?  Because it's mapped to the old value or index?

    # NOTE: I think visited (see commented out code below) should actually be a dict with the key being the destination and the value being the index i. (NOPE, dict keys have to be unique)
    # That way we can know which index holds the first instance of that destination, which is where we'll accumulate soil.
    #
    # NOTE: I have a concern about that extra soil being a problem. In the single-threaded version where each drop runs
    # one at a time the soil usually stays below 1 meter during any given drop's life from start to end.
    # But if we're running these in parallel it might accumulate several times as much as a single drop as the drops converge
    # which might affect the deposit rate or the character of the erosion.
    #
    # NOTE: We should consult the neighbors again. for n in neighbors: if n != -1: if the neighbor's destination is me: bring the soil to me, bring the drop to me, null out the soil at the neighbor, set that neighbor's dest to -1
    # The reason it is safe to also write to the neighbor and not just to me is that a vertex can only have ONE destination. So, there can't be a WRITE collision, and even if there's a READ collision it won't matter because
    # the other thread that is trying to read will only have two possible outcomes: Either it correctly reads the new state of -1 and skips it, or it incorrectly reads the old state, but that won't matter because it wasn't
    # possible for that other thread to return true for that other thread's vertex being the destination because the destination vert belonged to THIS thread only.
    #
    # Wait, if MY downstream neighbor wants to take my soil and move it to itself then if *I* want to take my upstream neighbor and move its soil to me then there could be a collision if both of us want to write to me...
    # Because we're writing to more than just ourselves..
    #
    # for i in prange drop_locs: zero out my carried soil.. for i in neighbors: sum their carried soil and add to me if I'm their destination.
    # Nope, won't work on its own because when one thread zeroes out the carried soil for a vert, then another thread trying to read that vert will read no carried soil which is wrong.
    # Basically we need to add another array.  One to read carried soil, one to write carried soil.  Which would be more optimal.. building this on the fly for each cluster of 6 neighbors and the discarding, or just 1 big array?
    # This might actually let us discard the idea of deduplication and just move the drops to their new locations and zero out their old locations and the deduplication will naturally happen as a result of that.

    # OLD
    # visited = set()
    # dupes = set()

    # for i in range(len(dests)):
    #     if dests[i] not in visited:
    #         visited.add(dests[i])
    #     else:
    #         dupes.add(dests[i])

    # dupes.remove(np.int32(-1))  # -1 is "no destination", so remove it.
    # visited = None

    # if len(dests < 45):
    #     print("drop_locs:", drop_locs)
    #     print("    dests:", dests)
    #     print("dupes:", dupes)

    ### Move water drops from dests into drop_locs

# Pseudocode ideas for running drops fully or at least partially in parallel without write collisions.
# Idea 0:
# Drops would deserve their own class, and each drop would have like drop.location, drop.destination, drop.carried_soil
# And we spawn a set{} or something full of drops and update their self.whatever and the heights arrays when they move.
# The hardest part might be making them play nice with numba.  It's also possible that instead of a class we just make an array or list of lists like [[64, 83, 0.12], [45, 859, 1.6]]
# where each [list] is a drop, and the items inside are [location, destination, carried_soil] but then this has a problem of mixed types, unless we store location and destination as floats as well
# and convert them to ints on the fly.  Or we could do what the finance industry does and turn the floats for carried_soil into ints by multiplying them by 100 or 1000 (depending on how many decimals we want to save)
# and then do the reverse when we want to turn them back into floats and use them.
# Importantly, if two drops have the same drop.location, we pick one to inherit the carried_soil from any others and then kill all others at that location.
#
# Idea 1:
# First we would initialize the drops and their starting locations similar to how we initialize drop_starts now.
# We don't need a separate array for drops; we can treat the index of the arrays as the ID for each drop.
# drop_locs = np.array or list(?) that contains current vertex location for each drop  [vertex 12, vertex 54, vertex 2000, etc.]
# dests = np.array or list(?) that contains the vertex destination for each drop, or -1; indexed the same as drop_locs
# carried_soil = np.array or list(?) that contains a float of carried soil; indexed the same as drop_locs
#
# Second, find the destination for each drop in drop_locs
# for i in prange(len(drop_locs)): # this part can be run in parallel because it's not changing anything
#     dests[i] = find_lowest(drop_locs[i], neighbors[drop_locs[i]], heights)
#
# Third, find out how much soil should be removed from heights and put that in carried_soil.
    # if dests[i] != -1:  # Is the pythonic way to say "if dests[i] is not -1"?  I'm not actually looking for math equality, I'm just using -1 as a symbol for 'no destination'. Python interns -1 so both are technically valid.
    #     do math to find slope, pick_up, etc. for each drop that has a destination
    #     add that amount to carried_soil[i] I think?
# Fourth, now we actually subtract that from write_buffer.  At this stage each vertex can still only have 1 drop so there should be
# no possible race condition impeding us from subtracting from each height in parallel.
# Also the carried_soil was initialized as 0.0 so it's safe to subtract 0 from a height,
# OR we could indent this 1 more time and put it inside the previous if statement since those are the only heights that we will subtract from.
#     write_buffer[drop_locs[i]] -= carried_soil[drop_locs[i]]
# Fifth, now comes the careful part.  We need to find any conflicts in dests; aka we need to deduplicate dests.
#     (You know what.. we could also just outright kill the duplicates without transferring their carried soil.)
#     (This would change the way the erosion works but would be a lot less complicated.)
# This might have to happen outside of the prange loop because now we're potentially in a many-to-one scenario where multiple drops have the same dest,
# and we need to take the carried soil from multiple and put them all in the same carried_soil bucket.
# [code???] Find any instance where a drop has the same destination as another drop.  A new list called duplicates? = [[1 , 2], [8, 20, 60]] ?  eugh, potentially different lengths
# for drop in duplicates:
#     carried_soil[duplicates[drop][0]] += carried_soil[duplicates[drop][1]] (or something)  # Take carried soil from any extras and give it to, say, simply the first drop in the list
#     carried_soil[duplicates[drop][0]] += carried_soil[duplicates[drop][2]] etc..
#     [duplicates[drop][1]] = 0
#     [duplicates[drop][2]] = 0 etc..
# Sixth, okay now we need to basically zero out the destination for any of the duplicates.
# for i in duplicates:  # Depends on the structure of duplicates
#     dests[i] = -1
# Seventh, now we can finally move the water drops to their destinations, because now there will only be 1 drop moving to 1 destination so no race condition should be possible
# for i in prange(len(drop_locs)):
#     if dests[i] != -1:
#         drop_locs[i] = dests[i]
# Eighth, calculate deposit
# Ninth, actually deposit and modify heights.  This should be safe because we previously culled any duplicates so it should be 1 deposit per drop location so this should be doable in parallel
# Tenth?  And finally the big question rears its head. Do we:
# 1) run a check on dests to see if every value is -1 and break the loop (maybe like if np.amax(dests) > -1), OR
# 2) pop/remove every -1 value in all 3 lists/arrays, thus changing their length, thus breaking the loop when they finally become empty?
# NOTE: I don't think we can actually change the length, or even do a view into the original array because a given vertex index could be higher than the length of the shortened array which will break indexing.
#
# Idea 2:
# Instead of a list or array that only contains active drops, make one that is the same len() as vertices, and track if each vertex has a drop or not.
# So the whole thing would be initialized as [-1, -1, -1, -1, -1] etc. and then we initialize the drops so it would look like
# [-1, -1, some_id, some_id, -1], but what is some_id?  That implies that there is another indexed list or array that actually contains the drops or something...
# Maybe this is more related to idea 0 and some_id would actually be the drop object.  Idea 0 needs a step where it checks to see if a vert has more than 1 drop.
#
# if drops[i] already has a drop (like, drop[i] != -1?):
#     take soil from current drop and give it to that drop
#     then kill this drop

# =========================

# Idea inspired by Axel Paris' description of aggregating the influence of each neighbor, with a write buffer to avoid race conditions.
# https://perso.liris.cnrs.fr/aparis/public_html/posts/terrain_erosion.html
# https://perso.liris.cnrs.fr/aparis/public_html/posts/terrain_erosion_2.html
def erode_terrain6(nodes, neighbors, heights, num_iter=1, snapshot=False):
    print("Starting terrain erosion...")
    if num_iter <= 0:
        num_iter = 1
        print(" ERROR: Cannot have less than 1 iteration.")

    write_buffer = np.ones_like(heights, dtype=np.float64)

    for i in range(num_iter):
        print("  Erosion pass:", i+1,"of", num_iter)
        erosion_iteration6(nodes, neighbors, heights, write_buffer)
        # Swap the read and write buffers
        # NOTE: The reason this had no perf gain might be because we're doing "else: w_buff[i] = r_buff[i]" inside the iteration.
        # Or it's because Numba is incredibly fickle.  Re-running the comparison again showed a substantially worse performance
        # for the old method and a better performance for the tuple swap.
        heights, write_buffer = write_buffer, heights
#        for x in prange(len(write_buffer)):
#            heights[x] = write_buffer[x]
        if snapshot:
            save_snapshot(heights, i)

    # For odd number of iterations, ensure buffers get swapped back at the end
    # NOTE: Does this mean snapshots actually lag behind by 1 iteration?
    if num_iter % 2 != 0:
        heights, write_buffer = write_buffer, heights

    if len(neighbors) < 43:
        print("New heights:")
        print(write_buffer)


@njit(cache=True, parallel=True, nogil=True)               # Only erode the current vertex based on its lowest neighbor.
def erosion_iteration6(verts, neighbors, r_buff, w_buff):  # This produces weird pentagon rings around the 12 vertices with connectivity 5.
    # simple_constant = 0.05
    simple_constant = 0.0005

    # For every vertex index
    for i in prange(len(verts)):
        # random.seed(i)  # This is suuuper slow
        # Aggregate amount by which to change height
        amt = 0.0

        dest = find_lowest(i, neighbors[i], r_buff)  # Find lowest neighbor

        # NOTE: Something area based instead of distance based would be better.

        ##### Attempt 01
        # if dest != -1:
        #     # d = calc_distance(verts[i], verts[dest])
        #     amt -= simple_constant * calc_distance(verts[i], verts[dest])# * random.random()
        # w_buff[i] = r_buff[i] + amt

        ##### Attempt 02
        # if dest != -1:
        #     # d = calc_distance(verts[i], verts[dest])
        #     amt -= simple_constant * calc_distance(verts[i], verts[dest])# * random.random()
        #     w_buff[i] = r_buff[i] + amt
        # else:
        #     w_buff[i] = r_buff[i]

        ##### Attempt 03 flattens things out very fast but doesn't accidentally dig lower
        if dest != -1:
            # d = calc_distance(verts[i], verts[dest])
            # NOTE: Because distance is in meters this is doing 0.0005 * ~22,000 = ~11 meters at a time!
            # So basically what this is doing is only lowering verts whose lowest neighbor are more than ~11 meters lower than themselves
            amt -= simple_constant * calc_distance(verts[i], verts[dest])# * random.random()
            new = r_buff[i] + amt
            if new < r_buff[dest]:  # If new value is lower than lowest neighbor.
                w_buff[i] = r_buff[i]  # Change nothing. Keep current vert at existing height. Current vert can't become lower than lowest neighbor.
                # w_buff[i] = r_buff[dest]  # Bring current vert down to lowest neighbor but no lower
            else:
                w_buff[i] = new  # Lower the current vert.
        else:
            w_buff[i] = r_buff[i]

    # NOTE: Consider breaking into more steps? The virtual pipes technique makes an extra 'array' of flow fields before actually moving things.
    # For erode_terrain6 that's more advanced than what we need but the extra step could be something like an array of how much sediment is at each vert,
    # then check available sediment at each neighbor in a subsequent step.

# =========================

# Triangle based erosion as opposed to vertex based erosion.
def erode_terrain7(cells):
    print("Not implemented yet")

    # for c in cells:
    #     for i, j in neighbor_pairs:
    #         aggregates[i] += elevation[j]
    #         aggregates[j] += elevation[i]
# In the end, you divide by 2 because you counted each contribution twice. Of course, you wouldn't use explicit Python loops, but np.add.at or npx.sum_at.
# https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
# https://github.com/nschloe/npx#sum_atadd_at
# NOTE: Numba does not yet support the "at" ufunc.


# =========================

# Observation 1: At higher levels of iterations (like 10+)
#   some line artifacts appear around the 12 verts with connectivity 5.
#   But that's far more iterations than should be used.

# Observation 2: The averaging effect becomes much weaker when the dvisions
#   increase because each vertex becomes much closer to each other vertex.
#   We should build an improved function that can average over a larger area.
#   (more than just the immediate 5 or 6 neighbors) (might need KD Tree)
#   I'd also like the ability to control the strength of the averaging, sort of
#   like setting the visibility of a layer in Photoshop to [0 to 100] percent
#   so that the below layer (original heights) can peek through.

# Observation 3: After implementing a distance-weighted average function
#   the line artifacts from observation 1 are still present. This leads me to
#   suspect that it is actually caused by the change in edge flow of the
#   triangles at the 'seams' connecting the 12 original vertices. The edge
#   artifacts seem effectively identical for the simple average method
#   and the weighted average method. And really since we are averaging
#   heights and not the actual vertex coordinates I don't think heights should
#   even be able to cause such an issue so it would have to be something else.
#   It also continues to happen at twice the divisions (320 --> 640)
#   and also doesn't appear to affect the exported height png so it has to be
#   something to do with the mesh itself like normal angle/smoothing.
def average_terrain1(tris, height, num_iter=1, snapshot=False):
    """Average terrain heights in place."""
    print("Averaging terrain...")
    if num_iter <= 0:
        num_iter = 1  # TODO: Better error handling.
        print(" ERROR: Cannot have less than 1 iteration.")

    neighbors = build_adjacency(tris)

    for i in range(num_iter):
        print("  Pass:", i+1,"of", num_iter)
        avg_terrain_iteration1(neighbors, height)

@njit(cache=True, parallel=True, nogil=True)
def avg_terrain_iteration1(nbrs, r_buff):
    """For each vertex, set its height to the average of its neighbors' heights,
    ignoring its own height (average only ring 1, excluding center vertex).
    """
    # NOTE: This is a little weird. We're doing reads from the copy and then doing writes into the original.
    # On the plus side this saves us from needing a copying step at the end.
    height_buffer = np.copy(r_buff)

    for v in prange(12):
        new = 0.0
        for n in nbrs[v][:5]:
            new += height_buffer[n]
        r_buff[v] = new / 5

    for v in prange (12, len(nbrs)):
        new = 0.0
        for n in nbrs[v]:
            new += height_buffer[n]
        r_buff[v] = new / 6


# This version is a little more crisp than averaging only ring 1.
# (Which makes sense because taking the average of ring 1 only is
# effectively discarding some data, making it less accurate.)
def average_terrain2(tris, height, num_iter=1, snapshot=False):
    """Average terrain heights in place."""
    print("Averaging terrain...")
    if num_iter <= 0:
        num_iter = 1  # TODO: Better error handling.
        print(" ERROR: Cannot have less than 1 iteration.")

    neighbors = build_adjacency(tris)

    for i in range(num_iter):
        print("  Pass:", i+1,"of", num_iter)
        avg_terrain_iteration2(neighbors, height)

@njit(cache=True, parallel=True, nogil=True)
def avg_terrain_iteration2(nbrs, r_buff):
    """For each vertex, set its height to the average of its own and its ring 1 neighbors' heights.
    """
    # NOTE: This is a little weird. We're doing reads from the copy and then doing writes into the original.
    # On the plus side this saves us from needing a copying step at the end.
    height_buffer = np.copy(r_buff)

    for v in prange(12):
        new = 0.0
        for n in nbrs[v][:5]:
            new += height_buffer[n]
        new += height_buffer[v]
        r_buff[v] = new / 6

    for v in prange (12, len(nbrs)):
        new = 0.0
        for n in nbrs[v]:
            new += height_buffer[n]
        new += height_buffer[v]
        r_buff[v] = new / 7


def average_terrain_weighted(verts, tris, height, num_iter=1, snapshot=False):
    """Average terrain heights in place using a distance weighted average.
    For each vertex, set its height to the distance weighted average of ring 1
    around that vertex, not including the height of the center vertex.
    """
    print("Averaging terrain...")
    if num_iter <= 0:
        num_iter = 1  # TODO: Better error handling.
        print(" ERROR: Cannot have less than 1 iteration.")

    neighbors = build_adjacency(tris)

    time_start = time.perf_counter()
    distances = build_distances(verts, neighbors)
    time_end = time.perf_counter()
    print(f"  Time to build vertex neighbor distances: {time_end - time_start :.5f} sec")

    for i in range(num_iter):
        print("  Pass:", i+1,"of", num_iter)
        avg_terrain_weighted_iteration(verts, neighbors, distances, height)

@njit(cache=True, parallel=True, nogil=True)
def build_distances(verts, nbrs):
    """Find distances to each vertex's neighbors."""
    # TODO: Is it correct for the first 12 verts to have positive 1 as a distance for the 6th position?
    # The only place that is using this function uses an array slice that ignores the 6th position but
    # this function may have other uses in the future where that might be a problem. Use -1 instead?
    # np.full((int(len(some_array)), 6), -1, dtype=np.some_dtype)
    result = np.ones_like(nbrs, dtype=np.float64)

    for v in prange(len(verts)):
        for i, n in enumerate(nbrs[v]):
            result[v][i] = calc_distance(verts[v], verts[n])
    return result

@njit(cache=True, parallel=True, nogil=True)
def avg_terrain_weighted_iteration(verts, nbrs, dists, r_buff):
    height_buffer = np.copy(r_buff)

    # TODO: Seeing that the inverse distance weighted code is being used in at least 3 places
    # it should probably be broken out into its own function, likely structured to take 1
    # array row at a time so that we can pass in slices like dists[v][:5] for the first 12 verts.
    for v in prange(12):
        sd = np.sum(dists[v][:5])  # Sum of distances
        # Add a tiny amount to each i to (hopefully) prevent NaN and div by 0 errors
        ws = np.array([1 / ((i+0.00001) / sd) for i in dists[v][:5]], dtype=np.float64)  # weights
        t = np.sum(ws)  # Total sum of weights
        iw = np.array([i/t for i in ws], dtype=np.float64)  # Inverted weights
        value = np.sum(np.array([height_buffer[nbrs[v][i]]*iw[i] for i in range(len(iw))]))

        r_buff[v] = value

    for v in prange(12, len(nbrs)):
        sd = np.sum(dists[v])  # Sum of distances
        # Add a tiny amount to each i to (hopefully) prevent NaN and div by 0 errors
        ws = np.array([1 / ((i+0.00001) / sd) for i in dists[v]], dtype=np.float64)  # weights
        t = np.sum(ws)  # Total sum of weights
        iw = np.array([i/t for i in ws], dtype=np.float64)  # Inverted weights
        value = np.sum(np.array([height_buffer[nbrs[v][i]]*iw[i] for i in range(len(iw))]))

        r_buff[v] = value

# =========================

# Fast Hydraulic Erosion Simulation and Visualization on GPU by Mei et al.
def erode_via_pipes(verts, tris, height, num_iter=1, snapshot=False):
    print("Not implemented yet")
    # Step 1: Add water

    # Step 2: Flow water
    # Inflow/Outflow flux calculation
    # Update water volume/height
    # Velocity field calculation

    # Step 3: Erosion-deposition
    # Dissolve some sediment
    # Deposit some sediment

    # Step 4: Transport sediment
    # Move suspended sediment with the velocity field

    # Step 5: Evaporation


@njit(cache=True, parallel=True, nogil=True)
def erode_via_pipes_iteration1(verts, nbrs, r_buff):
    print("Not implemented yet")


# Based on Boris Shishov's implementation
# https://github.com/bshishov/UnityTerrainErosionGPU
# https://github.com/bshishov/UnityTerrainErosionGPU/blob/master/Assets/Shaders/Erosion.compute
def erode_field2(verts, tris, height, num_iter=1, snapshot=False):
    print("Not implemented yet")
    # Step 1: Add water

    # Step 2: Compute flux field

    # Step 3: Apply flux

    # Step 4: Erosion-deposition
    # Also evaporation

    # Step 5: Transport sediment

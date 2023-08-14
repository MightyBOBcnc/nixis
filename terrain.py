"""Noise generators and other basic terrain functions."""

import time
import numpy as np
from numba import njit, prange
import opensimplex as osi
import cfg
# pylint: disable=not-an-iterable
# pylint: disable=line-too-long


@njit(cache=True, parallel=True, nogil=True)
def sample_noise(verts, perm, pgi, n_roughness=1, n_strength=0.2, radius=1):
    """Sample a simplex noise for given vertices."""
    elevations = np.ones(len(verts), dtype=np.float64)

    rough_verts = verts * n_roughness

    for v in prange(len(rough_verts)):
        elevations[v] = osi.noise3d(rough_verts[v][0], rough_verts[v][1], rough_verts[v][2], perm, pgi)

    # print(" Pre-elevations:")
    # print(elevations)
    # print(" min:", np.amin(elevations))
    # print(" max:", np.amax(elevations))
    # NOTE: Adding +1 to elevation moves negative values in the 0-1 range.
    # Multiplying by *0.5 drags any values > 1 back into the 0-1 range.
    return (elevations + 1) * 0.5 * n_strength * radius  # NOTE: I'm not sure if multiplying by the radius is the proper thing to do in my next implementation.
    # return elevations


def sample_octaves(verts, elevations, perm, pgi, n_octaves=1, n_init_roughness=1.5, n_init_strength=0.4, n_roughness=2.0, n_persistence=0.5, world_radius=1.0):
    """Sample octaves of noise and combine them together."""
    if elevations is None:
        elevations = np.zeros(len(verts), dtype=np.float64)
    n_freq = n_init_roughness  # Frequency
    n_amp = n_init_strength  # Amplitude
    # In my separate-sampling experiment, rough/strength pairs of (1.6, 0.4) (5, 0.2) and (24, 0.02) were good for 3 octaves
    # The final 3 results were added and then multiplied by 0.4
    for i in range(n_octaves):
        print(f"\r  Octave {i+1}..", flush =True, end ='')
        time_start = time.perf_counter()
        elevations += sample_noise(verts, perm, pgi, n_freq / world_radius, n_amp / world_radius, world_radius)
        n_freq *= n_roughness
        n_amp *= n_persistence
        time_end = time.perf_counter()
        print(f" {time_end - time_start :.5f} sec")


    # time_nmin = time.perf_counter()
    emin = np.amin(elevations)
    # time_nmax = time.perf_counter()
    emax = np.amax(elevations)
    # time_end = time.perf_counter()
    # print(f"  Time to find numpy min: {time_nmax - time_nmin :.5f} sec")
    # print(f"  Time to find numpy max: {time_end - time_nmax :.5f} sec")
    print("  Combined octaves min:", emin)
    print("  Combined octaves max:", emax)
    return elevations

@njit(cache=True, parallel=True, nogil=True)
def make_bool_elevation_mask(height, mask_elevation, mode):
    """Create a mask from an array of heights and a given elevation.
    height - Input heights.
    mask_elevation - The desired elevation cutoff point. (Inclusive)
    mode - A string for whether you wish to create the mask for values "above" or "below" mask_elevation.
    """
    # NOTE: Some other functions may not like that a dtype of bool_ stores "True" or "False" instead of numbers.
    # Keep an eye on that. We can always switch to int8 or uint8 and store 0 or 1 if we need to.
    mask = np.zeros(len(height), dtype=np.bool_)

    # NOTE: There's probably a more numpy-centric way to do this, like np.where()
    if mode == "below":
        for h in prange(len(height)):
            if height[h] <= mask_elevation:
                mask[h] = 1
    if mode == "above":
        for h in prange(len(height)):
            if height[h] >= mask_elevation:
                mask[h] = 1

    return mask

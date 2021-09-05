"""Module of climate functions."""
import numpy as np
from numba import njit, prange
from util import xyz2latlon, rescale

# ToDo: Modify temp based on altitude.  (Partially done; basic method is in place but need temps to change on an absolute scale, not a relative one from lowest to highest point.)
# ToDo: Make arrays for temperature at various altitudes? (i.e. not just at ground level, but also like the stratosphere etc.) https://scied.ucar.edu/learning-zone/atmosphere/change-atmosphere-altitude
# ToDo: Also establish the latitudes of the "tropics" bands of tempreatures (low, mid, high latitudes; tropics, subtropics, polar)
# ToDo: Factor in axial tilt (tropic of cancer, tropic of capricorn, etc.)  NOTE: latitude bands should probably be defined with a GLOBAL, maybe a dict
# ToDo: And due to axial tilt, take a time of year as an input to modify values (summer temps are different from winter temps)
#       The time of year should be set up that it can be any arbitrary amount of time, not a set number like 12, so that we can have planets with any orbital period, and cultures with arbitrary calendar increments (3 months, 60 weeks, 30 months, whatever)
# Examples:
# Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
# -26 -38 -50 -53 -53 -55 -56 -55 -55 -47 -35 -25 South Pole average high in C
# -30 -43 -57 -61 -61 -63 -64 -63 -63 -53 -39 -29 South Pole average low in C
# -15  -9  -4   6  15  21  22  18  12   2  -9 -14 North Pole average high in C
# -21 -17 -15  -4   4  11  13  10   4  -3 -15 -20 North Pole average low in C
# ToDo: Solar insolation. Eventually this is where the power of incoming solar radiation would be used to determine the energy and therefore the temperature (when combined with gas composition, albedo, black body radiation.. and other stuff..)
#   (I think for that part we can probably get away with a much-simplified calculation that might be for the full planet average or only for horizontal bands instead of per vertex.)
# ToDo: Climate/Biome classifications, e.g.  Köppen-Geiger, Trewartha, Holdridge, Whittaker. (I want to support multiple classification systems.)
#   This might prove helpful, although it's for processing images: https://github.com/MightyBOBcnc/speculative-koppen Maybe it can be adapted to icosphere, and more than 2 months.
#   https://en.wikipedia.org/wiki/Climate_classification and https://en.wikipedia.org/wiki/Biome
#   worldengine might also have some useful code in whatever is the most up-to-date fork. https://github.com/MightyBOBcnc/worldengine
#
# https://earthobservatory.nasa.gov/features/Water/page2.php
#   "the total amount of water vapor in the atmosphere remains approximately the same over time.
#    However, over the continents, precipitation routinely exceeds evaporation, and conversely, over the oceans, evaporation exceeds precipitation."
#
# ToDo: Not just rain, but frozen types of precipitation as well, and mixes (snow, sleet, etc.)
# ToDo: Not just evaporation, not just transpiration from plants, but also sublimation in colder regions.
# ToDo: Eventually perhaps take volcanic emissions into account.
# ToDo: Maybe an air quality index?  e.g. for tracking saharan-like dust storms, volcanoes, and forest fires?  lol, even pollen during spring.
# ToDo: I'd like to be able to determine areas on the planet that are most likely to experience severe weather (I'm mainly interested in thunderstorms but hurricanes, monsoons, extreme winds, and blizzards are also okay to do).
#   I imagine that a combination of humidity and temperature could produce a crude map like this.  Add in wind speed and atmospheric pressure to improve accuracy.  Maybe also a rate of exchange between air and land (precip and evap) with areas of high exchange.

@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp(verts, altitudes):
    """Assign starting surface temperatures based on equatorial distance and altitude."""
    # ToDo: Temps will only ever be in a range from approximately -60.0 to +60.0 C, so, what's a good dtype for that?  The world record low is approx -90 C, and the record high is approx +56 C.
    #       float16 could hold that but with little precision. float32 is probably the smallest safe choice.  I also may want room for higher and lower values than Earth produces.
    #       Consider also that Kelvin and Celsius have the same unit size; an increase of 1 K is identical to an increase of 1 C; the only difference is that C is offset by +273.15 from Kelvin values. (water freezes at 273.15 K and boils at 373.15 K)
    #       So given that I might want to someday expand the planet generator to non-Earthlike planets, or that I might need temperatures above/below Earth normals, maybe I should store them and calculate them in Kelvin.
    #       Also the fact that black body radiation is probably always done with kelvin and is tied to temperature and incoming/outgoing energy.
    surface_temps = np.zeros(len(verts), dtype=np.float32)

    # ToDo: Okay, so, temp fallof with altitude is tied to humidity; more humidity = slower falloff.
    # https://www.onthesnow.com/news/a/15157/does-elevation-affect-temperature This says temp falls off at about 9.8C per km in "dry" air, or 6C per km if humidity is 100% (NOTE: This wouldn't hold true for alien planets)
    # https://scied.ucar.edu/learning-zone/atmosphere/change-atmosphere-altitude Says average is 6.5C per km.  I think the consensus is 6.5C/km or 0.65 per 100 meters.
    # It's called "Lapse Rate" https://en.wikipedia.org/wiki/Lapse_rate and wiki says that the moist rate is more like 5C per km.. but the ICAO says an 'average' is 6.49 or 6.56C globally? Britannica suggests 6.5
    # The page also mentions dew points.  Pressure lapse rate is also a thing.
    alt_intensity = 1  # Controls strength of altitude contribution to temp.
    h2 = rescale(altitudes, 0, alt_intensity)

    for v in prange(len(verts)):  # pylint: disable=not-an-iterable
        dist_from_equator, _ = xyz2latlon(verts[v][0], verts[v][1], verts[v][2])

        # Linear dist_from_equator temp: 0 at poles, 1 at equator
        # surface_temps[v] = 1 - np.abs(dist_from_equator) / 90

        # Simple modification of temp based on altitude
#        surface_temps[v] = lerp1(1 - np.abs(dist_from_equator) / 90, 1, altitudes[v][0])
#        surface_temps[v] = 1 - (np.abs(dist_from_equator)/90 * (1 - altitudes[v][0]))
#        surface_temps[v] = 1 - (np.abs(dist_from_equator)/90 * -altitudes[v][0])
        surface_temps[v] = (1 - np.abs(dist_from_equator)/90) * (1 - np.abs(h2[v]))

    # return rescale(surface_temps, -60, 60)
    return surface_temps

# ToDo: Should it be called humidity or moisture? This is the water content of the air.
# ToDo: Do some book learnin' and find out if altitude modifies humidity. And temperature.
def assign_surface_humidity(verts, water_mask):
    # ToDo: What's the best dtype for humidity? Humidity can only range from 0-100 but I may want decimals. np.float16 might get me 1 or 2 decimal places, rounded. float32 gets me 5 or 6 decimal places, rounded. uint8 would be more than enough for int values.
    #       Once I have the climate system working in float32 it might be interesting to go back and change things out for float16 and see how much things change.
    humidity = np.zeros(len(verts), dtype=np.float32)

    return humidity

# Not sure if input1 or input2 is returned when mask is 0 (and the opposite as well; not sure which is returned when mask is 1)
@njit(cache=True)
def lerp1(input1, input2, mask):
    return (mask * input1) + ((1 - mask) * input2)  # convex combination

@njit(cache=True)
def lerp2(input1, input2, mask):
    # return input1 * (1 - mask) + input2 * mask
    return mask * input2 + (1 - mask) * input1


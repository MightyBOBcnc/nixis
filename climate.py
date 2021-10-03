"""Module of climate functions."""
import numpy as np
from numba import njit, prange
from util import xyz2latlon, rescale, load_settings
# pylint: disable=not-an-iterable

SBC = 5.670374419 * 10**-8  # Stefan-Boltzmann constant
# 0.00000005670374419

# Heat capacity in joules per kg per kelvin
# Sources:
# https://www.e-education.psu.edu/earth103/node/1005
# https://web.archive.org/web/20180210070117/http://www.kayelaby.npl.co.uk/general_physics/2_7/2_7_9.html
# https://theengineeringmindset.com/specific-heat-capacity-of-materials/
# https://www.researchgate.net/publication/245450023_Thermophysical_properties_of_seawater_A_review_of_existing_correlations_and_data
HEAT_CAPACITY = {
    "water": 4184, # At 25 C (262.15 k)
    "ice": 2008,
    "sea_water": 3989, # 3985 at 0 C, 3993 at 20 C.. but some sources say as low as 3850, or they say 3900...
    "average_rock": 2000,
    "wet_sand": 1500, # Defined as 20% water content
    "snow": 878,
    "dry_sand": 840,
    "vegetated_land": 830,
    "grass": 0,
    "soil": 0,
    "air": 700,
    "sandstone": 0
}

# Density in kg per meter^3
# Sources:
# https://www.e-education.psu.edu/earth103/node/1005
# https://www.eoas.ubc.ca/ubcgif/iag/foundations/properties/density.htm
# https://www.researchgate.net/post/How_we_can_determine_the_dry_sand_density
# http://www.antarcticglaciers.org/glaciers-and-climate/estimating-glacier-contribution-to-sea-level-rise/
# https://www.climate-policy-watcher.org/energy-balance/density-of-snow-and-ice.html
# https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
DENSITY = {
    "water": 1000,
    "ice": 917,
    "sea_water": 1027,
    "average_rock": 0,
    "wet_sand": 0, # Defined as 20% water content
    "snow": 100, # Fresh dry snow. Can go up to 200. Also, older compacted snow can be 400 to 500
    "dry_sand": 1600,
    "vegetated_land": 0,
    "grass": 0,
    "soil": 1522,
    "air": 1.2,
    "sandstone": 2323
}

# For Earth, average albedo of the whole planet is about 0.31 (although I've seen figures ranging from 0.29 to 0.31)
# Sources:
# https://nsidc.org/cryosphere/seaice/processes/albedo.html
# https://en.wikipedia.org/wiki/Albedo
# http://ponce.sdsu.edu/surface_albedo_and_water_resources.html Useful resource for aridity, moisture, runoff, and evaporation, too.
# 
ALBEDO = {
    "water": 0.06,
    "ice": 0.6, # Said to vary between 0.5 and 0.7 depending on age, compactedness, dirtiness
    "sea_water": 0.06,
    "average_rock":0,
    "wet_sand": 0, # Defined as 20% water content
    "snow": 0.8,
    "dry_sand": 0.4, # Using Wiki value for desert sand
    "vegetated_land": 0.17, # An average of the Wiki forest values
    "grass": 0.25, # Green grass
    "soil": 0.17,
    "air": -1, # Air is clear. Pretty sure it can't have an albedo, so, here's an impossible albedo value
    "sandstone": 0
}

# ToDo: Modify temp based on altitude.  (Partially done; basic method is in place but need temps to change on an absolute scale, not a relative one from lowest to highest point.)
# ToDo: Modify temp based on atmospheric pressure. e.g. the pressure changes with altitude; part of why death valley is hot is because it's below sea level. Mountain tops are colder with less air pressure.
# ToDo: Modify pressure based on temp. Wheeeee, it's cyclical. Higher temps cause pressure to lower as the air expands and vice versa. https://scied.ucar.edu/learning-zone/how-weather-works/highs-and-lows-air-pressure
# ToDo: Make arrays for temperature at various altitudes? (i.e. not just at ground level, but also like the stratosphere etc.) https://scied.ucar.edu/learning-zone/atmosphere/change-atmosphere-altitude
#       Temp at a given altitude in the air would be surface_temp - (lapse_rate * altitude), which is roughly accurate until you reach the tropopause at around 16 km.
#       The tropopause is coldest over the equator and warmest over the poles.  https://sciencing.com/tutorial-calculate-altitude-temperature-8788701.html
#       In the stratosphere things start to warm up again and T = -131 + (0.003 * altitude in meters) according to a NASA equation mentioned in the Sciencing article.
#       https://earthscience.stackexchange.com/questions/668/why-is-the-troposphere-8km-higher-at-the-equator-than-the-poles
# ToDo: Also establish the latitudes of the "tropics" bands of tempreatures (low, mid, high latitudes; tropics, subtropics, polar)
#   https://www.worldatlas.com/articles/what-is-the-effect-of-latitude-on-temperature.html
# ToDo: Factor in axial tilt (tropic of cancer, tropic of capricorn, etc.)  NOTE: latitude bands should probably be defined with a GLOBAL? maybe a dict?
#       Partially done for assign_surface_temp. There's axial tilt.  One thing that needs to be done is to consider, for the future, what day a year starts in relation to where in the orbit the planet is.
#       e.g. in the Western world we've standardized on the first day of the year being somewhat arbitrary compared to, say, a solstice or equinox. As a result, the year starts 10 days after the winter solstice and the 1st equinox is the 80th day of the year.
#       For the worldbuilders out there it would probably be helpful to give them a way to choose which day of the year is the first day.  Internally, I think Nixis will probably use an equinox as day 1 because that's the day with the easiest solar flux.
# ToDo: And due to axial tilt, take a time of year as an input to modify values (summer temps are different from winter temps)
#       The time of year should be set up that it can be any arbitrary amount of time, not a set number like 12, so that we can have planets with any orbital period, and cultures with arbitrary calendar increments (3 months, 60 weeks, 30 months, whatever)
# Examples:
# Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
# -26 -38 -50 -53 -53 -55 -56 -55 -55 -47 -35 -25 South Pole average high in C
# -30 -43 -57 -61 -61 -63 -64 -63 -63 -53 -39 -29 South Pole average low in C
# -15  -9  -4   6  15  21  22  18  12   2  -9 -14 North Pole average high in C
# -21 -17 -15  -4   4  11  13  10   4  -3 -15 -20 North Pole average low in C
# ToDo: Solar insolation. Eventually this is where the power of incoming solar radiation would be used to determine the energy and therefore the temperature (when combined with gas composition, albedo, black body radiation.. and other stuff..)
#   NOTE: The amount of watts per m^2 depends on the surface area, which depends on the radius. The radius defines a flat disc (cross section) of the planet pi*r^2 as viewed from the star, which gives the TSI "constant" which is then distributed over the surface 4*pi*r^2.
#   "the relative contribution of the atmosphere and the surface to each process (absorbing sunlight versus radiating heat) is asymmetric. The atmosphere absorbs 22 percent of incoming sunlight while the surface absorbs 47.
#    The atmosphere radiates heat equivalent to 58 percent of incoming sunlight; the surface radiates only 11 percent. In other words, most solar heating happens at the surface, while most radiative cooling happens in the atmosphere.""
#   https://earthobservatory.nasa.gov/features/EnergyBalance/page1.php
#   "The tropics (from 0 to 23.5° latitude) receive about 90% of the energy compared to the equator, the mid-latitudes (45°) roughly 70%, and the Arctic and Antarctic Circles (66.3°) about 40%." (This is verifiable by taking the cosine of the latitude.)
#   https://earthobservatory.nasa.gov/features/EnergyBalance/page2.php
#   "In the tropics there is a net energy surplus because the amount of sunlight absorbed is larger than the amount of heat radiated.
#    In the polar regions, however, there is an annual energy deficit because the amount of heat radiated to space is larger than the amount of absorbed sunlight."
#   https://earthobservatory.nasa.gov/features/EnergyBalance/page3.php
#   NOTE: "Insolation onto a surface is largest when the surface directly faces (is normal to) the sun. As the angle between the surface and the Sun moves from normal, the insolation is reduced in proportion to the angle's cosine"
#   https://www.e-education.psu.edu/earth103/node/1004
#   https://web.archive.org/web/20210301215506/https://www.sciencedirect.com/topics/earth-and-planetary-sciences/insolation
#   (I think for that part we can probably get away with a much-simplified calculation that might be for the full planet average or only for horizontal bands instead of per vertex.)
#   https://en.wikipedia.org/wiki/Solar_irradiance
#   https://en.wikipedia.org/wiki/Milankovitch_cycles  # Relevant to glaciation http://www.ces.fau.edu/nasa/module-3/temperature-trend-changes/causes-glaciation.php
#   Cool summers (generally from smaller axial tilt) are thought to allow winter snow to persist at higher latitudes, which allows snow to build up over geologic time.
#   https://earthobservatory.nasa.gov/features/Milankovitch/milankovitch_2.php
#   Note that while it might be tempting to even consider insolation on the day side vs night side of the planet, we're doing averages over long time periods which means day/night cycles are basically noise.
#   This class site has an awesome description (and equations) for an Energy Balance Model for the Earth's energy system.  https://www.ocean.washington.edu/courses/climate_extremes/Tutorial/Table_of_Contents.html
#   https://www.e-education.psu.edu/earth103/node/789
# * Polar areas get something like 40% of the insolation compared to equatorial parts of the Earth averaged over the year.
# ** Water has a specific heat that is, on average, 2.5 times higher than land (depends on the type of land; water is 2-5 times more)
# ** Land heats up and cools down faster than water.
# ** Air gets most of its temperature from IR radiation coming from the ground. Visible light passes through and heats the ground, which re-radiates in IR and heats the air.
# ** https://www.uou.ac.in/lecturenotes/science/MSCGE-19/Insolation,%20Atmospheric%20temperature%20and%20Heat%20Budget%20of%20the%20Earth.pdf
# * On and near the Solstices the summer pole actually receives MORE energy than anywhere else since there is 24 hours of daylight.
# ** http://www.applet-magic.com/insolation.htm
# * On an ordinary day, the insolation is proportional to the sine of the Sun's altitude. When the Sun is 30° above the horizon, the sunlight energy per square meter is half of what it is when the Sun is directly overhead. 
# ** https://svs.gsfc.nasa.gov/4466
# ToDo: Ocean water's huge volume and water's high specific heat capacity make it the dominant force in climate. The atmosphere is 'subservient' to the ocean.
#
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
# ToDo: Eventually perhaps take volcanic emissions into account.  (Note: If this is done over time, remember that there's probably much more volcanism on a young planet.)
# ToDo: Maybe an air quality index?  e.g. for tracking saharan-like dust storms, volcanoes, and forest fires?  lol, even pollen during spring.
# ToDo: I'd like to be able to determine areas on the planet that are most likely to experience severe weather (I'm mainly interested in thunderstorms but hurricanes, monsoons, extreme winds, and blizzards are also okay to do).
#   I imagine that a combination of humidity and temperature could produce a crude map like this.  Add in wind speed and atmospheric pressure to improve accuracy.  Maybe also a rate of exchange between air and land (precip and evap) with areas of high exchange.
# ToDo: Places near large bodies of water experience slower temperature changes because the nearby water is a heat sink. It takes more energy to heat/cool the water than land.
# ToDo: Orographic effect and rain shadows.  This will affect the erosion functions and the biome functions.
# ToDo: In the future, possibly consider the effect of solar cycles (e.g. sun spots and flare activity) and how that changes the watts/M^2 insolation on the planet over time?
# ToDo: Way the heck in the future maybe consider elliptical orbits.
# ToDo: Even further in the future, maybe take a shot at planets that are tidally locked to their host star.
# ToDo: For any calculation of the climate over time, consider that the brightness of a star changes over geologic time scales (e.g. our sun is 30% brighter than it was 4.5 billion years ago or like 3.3% every 500 million years).
#       Also the orbital distance of a planet changes as well (the star radiates its own mass away and Earth is slowly drifting a few meters away every year?).

@njit(cache=True)
def day2deg(year_length, day):
    """Convert a calendar day into degrees that the planet has orbited."""
    # return ((2*np.pi)/year_length) * day
    return (day/year_length) * 360.0  # ToDo: Should this be year_length-1 because we're counting from 0?

@njit(cache=True)
def deg2day(year_length, degrees):
    """Convert degrees that the planet has orbited into a calendar day."""
    return degrees * (year_length/360.0)  # ToDo: Should this be year_length-1 because we're counting from 0?

@njit(cache=True)
def calculate_seasonal_tilt(axial_tilt, degrees):
    """Find the seasonal tilt offset from axial tilt and orbit (in degrees)
    axial_tilt -- The planet's tilt. e.g. Earth's tilt is 23.44 degrees.
    degrees -- How far along is the planet in its orbit around its star?
    (between 0 and 360. 0/360 and 180 are equinoxes. 90 and 270 are solstices.)
    """
    return np.sin(degrees * np.pi/180) * axial_tilt

def calculate_tsi(star_radius, star_temp, orbital_distance):
    """Calculate the Total Solar Irradiance at the top of the atmosphere."""
    # Calculate the star's black-body radiation
    energy_at_sun = SBC * star_temp**4 * (4 * np.pi * star_radius**2)  # Energy at the star's surface
    energy_at_planet = energy_at_sun / (4 * np.pi * orbital_distance**2)  # Energy at the planet's orbit
    # energy_at_planet = (SBC * star_temp**4 * star_radius**2) / (orbital_distance**2)  # Simplified equation
    # energy_at_planet is at the top of the atmosphere. For Earth that's about 1370 watts per meter squared (W/m^2)
    return energy_at_planet

@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp(verts, altitudes, tilt, surf_watts):
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
    # https://link.springer.com/article/10.1007/BF02247093  Lapse rate also changes by time of year as temperatures change from axial tilt (which affect humidity)
    #
    #
    alt_intensity = 0  # Controls strength of altitude contribution to temp.
    h2 = rescale(altitudes, 0, alt_intensity)  # If heights are already given in absolute meters then we could multiply by the lapse rate (for values > sea level) without even doing a rescale.
    # Although afterward it might be best to divide by the world_radius to get these back into a 0-1 range? Because the issue here is that the heights would have absolute temperatures, but the
    # Latitude-based temps would be still in the -1 to +1 range, which makes them incompatible.  I guess what could be done is:
    # 1. Multiply absolute heights by the lapse rate in the h2 array.
    # 2. Get the latitude temps separately in -1 to +1 range in the surface_temps array.
    # 3. Rescale the surface_temps array to absolute values (for some arbitrary values...).
    # 4. Subtract h2 from surface_temps and return surface_temps.
    # But ideally the latitude temps should be derived from first principles, aka solar insolation and albedo

    for v in prange(len(verts)):
        lat, lon = xyz2latlon(verts[v][0], verts[v][1], verts[v][2])
        dist_from_equator = lat - tilt

        # Linear dist_from_equator temp: 0 at poles, 1 at equator
        # surface_temps[v] = 1 - np.abs(dist_from_equator) / 90

        # Simple modification of temp based on altitude
#        surface_temps[v] = lerp1(1 - np.abs(lat) / 90, 1, altitudes[v][0])
#        surface_temps[v] = 1 - (np.abs(lat)/90 * (1 - altitudes[v][0]))
#        surface_temps[v] = 1 - (np.abs(lat)/90 * -altitudes[v][0])

        # surface_temps[v] = (1 - np.abs(lat)/90) * (1 - np.abs(h2[v]))  # Linear falloff
        # surface_temps[v] = max((1 - np.abs(dist_from_equator)/90), 0.01) * (1 - np.abs(h2[v]))  # Linear falloff with axial tilt

        # surface_temps[v] = (1 - np.abs(dist_from_equator)/90)**2 * (1 - np.abs(h2[v]))  # Quadratic falloff? This looks way too cold.
        # surface_temps[v] = (1 - (np.abs(dist_from_equator)/90)**2) * (1 - np.abs(h2[v]))  # Quadratic falloff? V2. This actually kinda looks like Tenjix's quadratic fallof (wider warm bands) although my linear altitude falloff is messed up.
        # surface_temps[v] = max((1 - np.abs(dist_from_equator)/90)**2, 0.01) * (1 - np.abs(h2[v]))  # Quadratic? falloff with axial tilt

        # ==========
        # These two seem to be the best candidates. I am uncertain about what a correct, proper distribution on a sphere looks like as you leave the equator (linear or quadratic)
        # surface_temps[v] = (1 - np.abs(dist_from_equator)/90) - np.abs(h2[v])  # Linear falloff with axial tilt  NOTE: This might be a better way to handle altitude. Just subtract it from sea level temp at that location.
##        surface_temps[v] = (1 - (np.abs(dist_from_equator)/90)**2) - np.abs(h2[v])  # Quadratic falloff? V3.

##        surface_temps[v] = np.cos(np.abs(dist_from_equator) * np.pi/180) - np.abs(h2[v])  # This is similar to quadratic but has a wider cold band. According to wiki this is the proper falloff. The real question is, is the performance worse?
##        surface_temps[v] = max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0) - np.abs(h2[v]) # Quadratic is visually close enough to this that it might be worth using quadratic if it's faster than converting latitude degrees to radians, then doing cosine.
        # It should also be noted that I'm not sure what to do with areas that don't receive sun. IRL they get some heating from circulation but I'm not simulating that yet.
        # So as far as watts reaching the surface from the sun, it's all 0 and you can't go below that number, but in reality these locations do have heat capacity and heat from elsewhere, so I'm not sure how to convert this into a temperature.
        # Maybe I could give them their own fake temperature that is a falloff from the nearest lit areas, like a gaussian blur or gradient from lit areas to unlit areas. NOTE: Read more about the polar cells and how they transport heat.

        # Same thing as above but oceans can have 0.1 subtracted to make them cooler than the shore closest to them, (or the land could have 0.1 added to make it warmer) which looks like Tenjix's temperature map image in his thesis.
        # surface_temps[v] = max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0) - np.abs(h2[v]) if h2[v] > 0 else max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0) - 0.1
        surface_temps[v] = max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0) - np.abs(h2[v]) + 0.1 if h2[v] > 0 else max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0)

        # ToDo: Test splitting this into a variable for the equator falloff, square the variable, then set st[v] to that * 1-h2 because I want to see if that might run faster than doing it all on one line.
        # Something similar was true for the sample_noise function. It was faster to fill the array with a simple operation and then do multiplication at the end. Although I'm not sure if I have any simple constants to multiply against here.

    # return rescale(surface_temps, -60, 60)
    return surface_temps

@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp2(verts, altitudes, tilt, surf_watts):
    surface_temps = np.zeros(len(verts), dtype=np.float32)
    alt_intensity = 0  # Controls strength of altitude contribution to temp.
    h2 = rescale(altitudes, 0, alt_intensity)

    # tilt = 60

    for v in prange(len(verts)):
        # https://www.mathworks.com/matlabcentral/answers/123763-how-to-rotate-entire-3d-data-with-x-y-z-values-along-a-particular-axis-say-x-axis
        x = verts[v][0]*np.cos(tilt * np.pi/180) + verts[v][2]*np.sin(tilt * np.pi/180)
        z = verts[v][2]*np.cos(tilt * np.pi/180) - verts[v][0]*np.sin(tilt * np.pi/180)
        lat, lon = xyz2latlon(x, verts[v][1], z)

        # lat, lon = xyz2latlon(verts[v][0], verts[v][1], verts[v][2])
        dist_from_equator = lat - tilt
        # surface_temps[v] = max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0) - np.abs(h2[v]) + 0.1 if h2[v] > 0 else max(np.cos(np.abs(dist_from_equator) * np.pi/180), 0)

        surface_temps[v] = max(np.cos(np.abs(lat) * np.pi/180), 0) * max(np.cos(lon * np.pi/180), 0)

    return surface_temps

@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp3(verts, altitudes, tilt, surf_watts):
    surface_temps = np.zeros(len(verts), dtype=np.float32)
    alt_intensity = 0  # Controls strength of altitude contribution to temp.
    h2 = rescale(altitudes, 0, alt_intensity)

    # tilt = 60

    for v in prange(len(verts)):
        # https://www.mathworks.com/matlabcentral/answers/123763-how-to-rotate-entire-3d-data-with-x-y-z-values-along-a-particular-axis-say-x-axis
        x = verts[v][0]*np.cos(tilt * np.pi/180) + verts[v][2]*np.sin(tilt * np.pi/180)
        z = verts[v][2]*np.cos(tilt * np.pi/180) - verts[v][0]*np.sin(tilt * np.pi/180)
        lat, lon = xyz2latlon(x, verts[v][1], z)

        # lat, lon = xyz2latlon(verts[v][0], verts[v][1], verts[v][2])
        dist_from_equator = lat - tilt

        for i in range(-180, 180, 1):
            surface_temps[v] += max(np.cos(np.abs(lat) * np.pi/180), 0) * max(np.cos(i * np.pi/180), 0)

    return surface_temps

@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp4(verts, altitudes, tilt, surf_watts):
    surface_temps = np.zeros(len(verts), dtype=np.float32)
    alt_intensity = 0  # Controls strength of altitude contribution to temp.
    h2 = rescale(altitudes, 0, alt_intensity)

    # tilt = 60

    for v in prange(len(verts)):
        for i in range(-180, 180, 1):
        # for i in range(-180, -178, 1):
            # Rotate the planet
            x = verts[v][0]*np.cos(i * np.pi/180) - verts[v][1]*np.sin(i * np.pi/180)
            y = verts[v][0]*np.sin(i * np.pi/180) + verts[v][1]*np.cos(i * np.pi/180)
            z = verts[v][2]

            # Tilt the planet
            x = x*np.cos(tilt * np.pi/180) + z*np.sin(tilt * np.pi/180)
            # y = y
            z = z*np.cos(tilt * np.pi/180) - x*np.sin(tilt * np.pi/180)

            lat, lon = xyz2latlon(x, y, z)

            # lat, lon = xyz2latlon(verts[v][0], verts[v][1], verts[v][2])
            dist_from_equator = lat - tilt

            surface_temps[v] += max(np.cos(np.abs(lat) * np.pi/180), 0) * max(np.cos(i * np.pi/180), 0)

    return surface_temps

@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp5(verts, altitudes, tilt, surf_watts):
    surface_temps = np.zeros(len(verts), dtype=np.float32)
    alt_intensity = 0  # Controls strength of altitude contribution to temp.
    h2 = rescale(altitudes, 0, alt_intensity)

    # tilt = -90
    rotation = -180.0
    # rot_steps = 86400  # Good god this takes a long time.  A faster approximation would be to only calculate a single half-circle of points from pole to pole at fixed latitudes
    rot_steps = 360  # then do a lookup for each vert's latitude and interpolate the result.
    rot_amt = 360.0 / rot_steps

    for v in prange(len(verts)):
        # for i in range(-180, 180, 1):
        # for i in range(-120, 121, 1):
        # for i in range(0, 1, 1):
        for i in range(rot_steps):
            # Rotate the planet
            x = verts[v][0]*np.cos(rotation * np.pi/180) - verts[v][1]*np.sin(rotation * np.pi/180)
            y = verts[v][0]*np.sin(rotation * np.pi/180) + verts[v][1]*np.cos(rotation * np.pi/180)
            z = verts[v][2]

            # Rotate the planet
            # x = verts[v][0]*np.cos(i * np.pi/180) - verts[v][1]*np.sin(i * np.pi/180)
            # y = verts[v][0]*np.sin(i * np.pi/180) + verts[v][1]*np.cos(i * np.pi/180)
            # z = verts[v][2]

            # Tilt the planet
            a = x*np.cos(tilt * np.pi/180) + z*np.sin(tilt * np.pi/180)
            b = y
            c = z*np.cos(tilt * np.pi/180) - x*np.sin(tilt * np.pi/180)

            # lat, lon = xyz2latlon(x, y, z)
            lat, lon = xyz2latlon(a, b, c)

            # lat, lon = xyz2latlon(verts[v][0], verts[v][1], verts[v][2])
            dist_from_equator = lat - tilt

            surface_temps[v] += max(np.cos(np.abs(lat) * np.pi/180), 0) * max(np.cos(lon * np.pi/180), 0)  # Okay, so next step is to turn this into temperatures, and then after that subtract the land temps based on altitude.
            # NOTE: Also need to compare the distribution to the simple model in the first assign_surface_temps and see if it visually looks like the same distribution. (the numbers on the scale bar will be different, of course)
            rotation += rot_amt

    return surface_temps

# ToDo: Should it be called humidity or moisture? This is the water content of the air.  Pretty sure it's humidity.
# ToDo: Do some book learnin' and find out if altitude modifies humidity. And temperature.
# https://en.wikipedia.org/wiki/Properties_of_water
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


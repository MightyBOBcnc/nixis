"""Module of climate functions."""
import os
import sys
import numpy as np
from numba import njit, prange
import cfg
from util import xyz2latlon, latlon2xyz, rescale, load_settings, build_image_data, save_image
# pylint: disable=not-an-iterable
# pylint: disable=line-too-long

SBC = 5.670374419 * 10**-8  # Stefan-Boltzmann constant
# 0.00000005670374419

# TODO: Dict is incomplete. (e.g. missing materials, and materials whose value is "0" as a placeholder)
# Heat capacity in joules per kg per kelvin
# Sources:
# https://www.e-education.psu.edu/earth103/node/1005
# https://web.archive.org/web/20180210070117/http://www.kayelaby.npl.co.uk/general_physics/2_7/2_7_9.html
# https://theengineeringmindset.com/specific-heat-capacity-of-materials/
# https://www.researchgate.net/publication/245450023_Thermophysical_properties_of_seawater_A_review_of_existing_correlations_and_data
MAT_HEAT_CAPACITY = {
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

# TODO: Dict is incomplete. (e.g. missing materials, and materials whose value is "0" as a placeholder)
# Density in kg per meter^3
# Sources:
# https://www.e-education.psu.edu/earth103/node/1005
# https://www.eoas.ubc.ca/ubcgif/iag/foundations/properties/density.htm
# https://www.researchgate.net/post/How_we_can_determine_the_dry_sand_density
# http://www.antarcticglaciers.org/glaciers-and-climate/estimating-glacier-contribution-to-sea-level-rise/
# https://www.climate-policy-watcher.org/energy-balance/density-of-snow-and-ice.html
# https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
MAT_DENSITY = {
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

# TODO: Dict is incomplete. (e.g. missing materials, and materials whose value is "0" as a placeholder)
# For Earth, average albedo of the whole planet is about 0.31 (although I've seen figures ranging from 0.29 to 0.31)
# Sources:
# https://nsidc.org/cryosphere/seaice/processes/albedo.html
# https://en.wikipedia.org/wiki/Albedo
# http://ponce.sdsu.edu/surface_albedo_and_water_resources.html Useful resource for aridity, moisture, runoff, and evaporation, too.
#
MAT_ALBEDO = {
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

# TODO: Modify temp based on altitude.  (Partially done; basic method is in place but need temps to change on an absolute scale, not a relative one from lowest to highest point.)
# TODO: Modify temp based on atmospheric pressure. e.g. the pressure changes with altitude; part of why death valley is hot is because it's below sea level. Mountain tops are colder with less air pressure.
# TODO: Modify pressure based on temp. Wheeeee, it's cyclical. Higher temps cause pressure to lower as the air expands and vice versa. https://scied.ucar.edu/learning-zone/how-weather-works/highs-and-lows-air-pressure
# TODO: Make arrays for temperature at various altitudes? (i.e. not just at ground level, but also like the stratosphere etc.) https://scied.ucar.edu/learning-zone/atmosphere/change-atmosphere-altitude
#       Temp at a given altitude in the air would be surface_temp - (lapse_rate * altitude), which is roughly accurate until you reach the tropopause at around 16 km.
#       The tropopause is coldest over the equator and warmest over the poles.  https://sciencing.com/tutorial-calculate-altitude-temperature-8788701.html
#       In the stratosphere things start to warm up again and T = -131 + (0.003 * altitude in meters) according to a NASA equation mentioned in the Sciencing article.
#       https://earthscience.stackexchange.com/questions/668/why-is-the-troposphere-8km-higher-at-the-equator-than-the-poles
# TODO: Also establish the latitudes of the "tropics" bands of tempreatures (low, mid, high latitudes; tropics, subtropics, polar)
#   https://www.worldatlas.com/articles/what-is-the-effect-of-latitude-on-temperature.html
# TODO: Factor in axial tilt (tropic of cancer, tropic of capricorn, etc.)  NOTE: latitude bands should probably be defined with a GLOBAL? maybe a dict?
#       Partially done for assign_surface_temp. There's axial tilt.  One thing that needs to be done is to consider, for the future, what day a year starts in relation to where in the orbit the planet is.
#       e.g. in the Western world we've standardized on the first day of the year being somewhat arbitrary compared to, say, a solstice or equinox. As a result, the year starts 10 days after the winter solstice and the 1st equinox is the 80th day of the year.
#       For the worldbuilders out there it would probably be helpful to give them a way to choose which day of the year is the first day.  Internally, I think Nixis will probably use an equinox as day 1 because that's the day with the easiest solar flux.
# TODO: And due to axial tilt, take a time of year as an input to modify values (summer temps are different from winter temps)
#       The time of year should be set up that it can be any arbitrary amount of time, not a set number like 12, so that we can have planets with any orbital period, and cultures with arbitrary calendar increments (3 months, 60 weeks, 30 months, whatever)
#       This is partially started with the day2deg and deg2day functions.
# Examples:
# Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
# -26 -38 -50 -53 -53 -55 -56 -55 -55 -47 -35 -25 South Pole average high in C
# -30 -43 -57 -61 -61 -63 -64 -63 -63 -53 -39 -29 South Pole average low in C
# -15  -9  -4   6  15  21  22  18  12   2  -9 -14 North Pole average high in C
# -21 -17 -15  -4   4  11  13  10   4  -3 -15 -20 North Pole average low in C
# TODO: Solar insolation. Eventually this is where the power of incoming solar radiation would be used to determine the energy and therefore the temperature (when combined with gas composition, albedo, black body radiation.. and other stuff..)
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
# TODO: Ocean water's huge volume and water's high specific heat capacity make it the dominant force in climate. The atmosphere is 'subservient' to the ocean.
#
# TODO: Climate/Biome classifications, e.g.  Köppen-Geiger, Trewartha, Holdridge, Whittaker. (I want to support multiple classification systems.)
#   This might prove helpful, although it's for processing images: https://github.com/MightyBOBcnc/speculative-koppen Maybe it can be adapted to icosphere, and more than 2 months.
#   https://en.wikipedia.org/wiki/Climate_classification and https://en.wikipedia.org/wiki/Biome
#   worldengine might also have some useful code in whatever is the most up-to-date fork. https://github.com/MightyBOBcnc/worldengine
#   It should be noted that some (if not all?) of these classification systems require knowing the average temperature and precipitation across multiple snapshots during the year so you can compare winter to summer.
#   Therefore we'll need to store multiple arrays for each component, such as temp_Jan temp_Feb temp_Mar and precip_Jan precip_Feb etc.
#   or it could be one array for each component with elements for each snapshot like temps = [[Jan, Feb, Mar], [Jan, Feb, Mar], [Jan, Feb, Mar]] (where each [Jan, Feb, Mar] is snapshots for each vertex)
#
# https://earthobservatory.nasa.gov/features/Water/page2.php
#   "the total amount of water vapor in the atmosphere remains approximately the same over time.
#    However, over the continents, precipitation routinely exceeds evaporation, and conversely, over the oceans, evaporation exceeds precipitation."
#
# TODO: Not just rain, but frozen types of precipitation as well, and mixes (snow, sleet, etc.)
# TODO: Not just evaporation, not just transpiration from plants, but also sublimation in colder regions.
# TODO: When I start implementing plant growth and 'true color' maps, take into account that the pigment of the plants will likely respond to the spectrum and intensity of the star.
#   Some worlds will have green plants, others red, some black, etc. https://www.reddit.com/r/worldbuilding/comments/1dui0v/plants_on_other_worlds_reference_chart/
#   Additional reference for plant pigment color vs star spectrum: http://panoptesv.com/SciFi/ColorsOfAlienWorlds/AlienFields.php
# TODO: Eventually perhaps take volcanic emissions into account.  (NOTE: If this is done over time, remember that there's probably much more volcanism on a young planet.)
# TODO: Maybe an air quality index?  e.g. for tracking saharan-like dust storms, volcanoes, and forest fires?  lol, even pollen during spring.
# TODO: I'd like to be able to determine areas on the planet that are most likely to experience severe weather (I'm mainly interested in thunderstorms but hurricanes, monsoons, extreme winds, and blizzards are also okay to do).
#   I imagine that a combination of humidity and temperature could produce a crude map like this.  Add in wind speed and atmospheric pressure to improve accuracy.  Maybe also a rate of exchange between air and land (precip and evap) with areas of high exchange.
# TODO: Places near large bodies of water experience slower temperature changes because the nearby water is a heat sink. It takes more energy to heat/cool the water than land.
# TODO: Orographic effect and rain shadows.  This will affect the erosion functions and the biome functions.
# TODO: In the future, possibly consider the effect of solar cycles (e.g. sun spots and flare activity) and how that changes the watts/M^2 insolation on the planet over time?
# TODO: Way the heck in the future maybe consider elliptical orbits.
# TODO: Even further in the future, maybe take a shot at planets that are tidally locked to their host star.
# TODO: For any calculation of the climate over time, consider that the brightness of a star changes over geologic time scales (e.g. our sun is 30% brighter today than it was 4.5 billion years ago or like 3.3% every 500 million years).
#       Also the orbital distance of a planet changes as well (the star radiates its own mass away and Earth is slowly drifting a few meters away every year? The drifting away might be momentum/gravity related, not the mass change).
#       https://www.nature.com/articles/s41586-021-03873-w
#       https://www.space.com/venus-never-habitable-no-oceans
#       "in the early days of the solar system, our star was just 70% as luminous as it is now."
#       This same study notes that Earth could have become a steam house if the sun had been brighter. Which illustrates something about Nixis' assumptions: It doesn't simulate formation of the crust, or the formation of oceans, it just assumes they exist.
#       https://www.space.com/19118-early-earth-atmosphere-faint-sun.html
#       And the gas composition changes, too, particularly if life starts spewing oxygen, of course, which changes the greenhouse strength.
# TODO: "Vapor Pressure Deficit" seems like something that might be easy to approximate: https://phys.org/news/2021-11-increasingly-frequent-wildfires-linked-human-caused.html
# TODO: Glaciation causes sea level to drop as water gets locked up in ice.  And the reverse is true when glaciers melt.

@njit(cache=True)
def day2deg(year_length, day):
    """Convert a calendar day into degrees that the planet has orbited."""
    # return ((2*np.pi)/year_length) * day
    return (day/year_length) * 360.0  # TODO: Should this be year_length-1 because we're counting from 0?

@njit(cache=True)
def deg2day(year_length, degrees):
    """Convert degrees that the planet has orbited into a calendar day."""
    return degrees * (year_length/360.0)  # TODO: Should this be year_length-1 because we're counting from 0?

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
@njit(cache=True)
def find_nearest(arr, num):
    """Find the index of the number nearest to the provided number."""
    # idx = (np.abs(arr - num)).argmin()
    # return arr[idx]  # Returns the value of the nearest number
    return (np.abs(arr - num)).argmin()  # Returns the index

# TODO: I wonder if a triangle wave would be more appropriate than a sine wave? A circular orbit is (effectively) a constant speed but a sine wave isn't linear.
# One of the pages I read about tilt in the context of a 90 degree tilted axis said the transition of the solar zenith would be essentially a continuous linear motion
# from pole to pole and back again. On the other hand, all the annual insolation chart graphics seem to approximate a sine more than a linear spike.
# https://stackoverflow.com/questions/12332392/triangle-wave-shaped-array-in-python
# Oh, I know, I should graph my 1980 reference spreadsheet at a few given latitudes and see how that looks.
# (I did this and they're sine-like; I should really output my arrays into a csv so I can graph Nixis data directly, or maybe pyvista or scipy line graphing.)
@njit(cache=True)
def calc_seasonal_tilt(axial_tilt, degrees):
    """Find the seasonal tilt offset from axial tilt and orbit (in degrees)
    axial_tilt -- The planet's tilt. e.g. Earth's tilt is 23.44 degrees.
    degrees -- How far along is the planet in its orbit around its star?
    (between 0 and 360. 0/360 and 180 are equinoxes. 90 and 270 are solstices.)
    """
    # NOTE: IRL the tilt of a planet doesn't actually change as it orbits.
    # What does change is the *relative* angle of incoming sunlight.
    return np.sin(degrees * np.pi/180) * axial_tilt

def calc_tsi(star_radius, star_temp, orbital_distance):
    """Calculate the Total Solar Irradiance at the top of the atmosphere.
    star_radius -- in km
    star_temp -- in kelvin
    orbital_distance -- in km"""
    # Calculate the star's black-body radiation
    energy_at_sun = SBC * star_temp**4 * (4 * np.pi * star_radius**2)  # Energy at the star's surface
    energy_at_planet = energy_at_sun / (4 * np.pi * orbital_distance**2)  # Energy at the planet's orbit
    # energy_at_planet = (SBC * star_temp**4 * star_radius**2) / (orbital_distance**2)  # Simplified equation
    # energy_at_planet is at the top of the atmosphere. For Earth that's about 1370 watts per meter squared (W/m^2)
    return energy_at_planet

def calc_planet_equilibrium(flux, albedo, radius=1):
    """Calculate the equilibrium temperature for a whole planet."""
    tsi = flux

    # https://earthobservatory.nasa.gov/features/EnergyBalance/page1.php
    # TSI is at the top of the atmosphere. For Earth that's about 1370 watts per meter squared (W/m^2)
    # Because only half of the planet is lit, this immediately cuts that number in half.
    # Because a sphere isn't evenly lit (cosine falloff with latitude and longitude), this is cut in half again.
    # So, when averaged over the whole planet the incoming energy is 0.25 * TSI or ~ 342.5 W/m^2
    # But in addition to that some simply bounces away because of albedo. About 31% bounces away. 342 * (1 - 0.31) or 236.325 W/m^2 (NOTE: I've seen sources say albedo is 29% to 31%)
    # 22% gets absorbed by the atmosphere by water vapor, dust, and ozone, and 47% gets absorbed by the surface. (so of the 69% that doesn't bounce, about 31.8% is absorbed in the atmosphere and 68.2% by the surface)

    earth_radius = 6378100.0  # NOTE: Manually keying this in for now because something is weird/wrong with visualizing the temperatures on a full-sized planet at the moment so I'm visualizing at radius 1.0 for the time being.
    # NOTE: The visualization issue might possibly have been fixed when I fixed the NaN problem with xyz2latlon.

    planet_cross_section = np.pi * earth_radius**2
    planet_surface_area = 4 * np.pi * earth_radius**2  # Sunlight only falls on half of the surface area, however

    # TODO: energy at the cross section, energy reduced by albedo, and maybe energy that bounces off the atmosphere? not sure if that's the same thing.  or maybe I meant absorbed by the atmosphere so it doesn't reach ground.
    # For basic temp assignment, I could start by calculating the airless body temperature and then just multiply by a fudge factor for greenhouse effect. (e.g. earth is like 34 C higher than it would be with no atmospheric blanket)
    # Basic energy balance in/out
    total_energy = tsi * planet_cross_section  # Total energy of the planet cross-section, in W
    energy_in = tsi * 0.25 * (1 - albedo)  # Energy in after factoring out albedo bounce and spherical geometry, in W/m^2
    atmosphere_watts = energy_in * 0.318  # After factoring out albedo bounce, this is how much the atmosphere absorbs in W/m^2
    surface_watts = energy_in * 0.682  # After factoring out albedo bounce, this is how much the surface absorbs in W/m^2

    # These values would probably be derived from the gas/aerosol composition of the atmosphere (likely including water vapor but possibly not clouds as that's albedo?)
    # (I think the open source "Thrive" by Revolutionary Games does this; e.g. calculating light blockage via its wavelength in nanometers vs size of molecules)
    # Amount of incoming shortwave light frequencies that makes it down through the atmosphere
    shortwave_transmittance = 0.9
    # Amount of outgoing longwave IR frequencies that makes it out through the atmosphere
    # As you can see, Earth's atmosphere blocks most outgoing IR radiation from the ground.
    # It is absorbed by the atmosphere and then re-radiated away.
    longwave_transmittance = 0.16

    # There is a mismatch with what's written on the linked GHG1 page below
    # and what needs to be done. There is some poor/sloppy phrasing.
    # DON'T raise this to the power of 0.25. What looks like an earlier
    # version confirms this: http://homework.uoregon.edu/pub/class/es202/atm.html
    ground_flux = tsi * ((1 + shortwave_transmittance) / (1 + longwave_transmittance))  # Simple way to fudge atmospheric greenhouse blanket effect

    # Okay this equation here was a bit hard to figure out but its the
    # average surface temp for the whole planet without atmospherics.
    # The source is at the bottom of the following page:
    # http://homework.uoregon.edu/pub/class/es202/GRL/ghg1.html
    avg_surface_temp_unmodified = (tsi / (4*SBC))**0.25 * (1 - albedo)**0.25
    # Same but with atmospheric fudge plugged in
    avg_surface_temp_greenhouse = (ground_flux / (4*SBC))**0.25 * (1 - albedo)**0.25

    print("--                TSI is:", tsi)
    print("--Cross-sectional energy:", total_energy)
    print("--Ground flux is:        ", ground_flux)
    print("--Original surface temp: ", avg_surface_temp_unmodified)
    print("--Modified surface temp: ", avg_surface_temp_greenhouse)

def calc_water_equilibrium(flux):
    """Calculate the equilibrium temperature for a block of water."""
    edge_length = 1.0  # In meters
    volume = edge_length**3  # X cubic meters  # NOTE: Consider whether it would be more relevant to do this with circles with an area of 1 meter squared, and/or cylinders of material.
    # volume = edge_length**2  # For water with a depth of only 1 meter
    # surface_area = edge_length**2  # Allow only 1 side of the cube to radiate energy
    surface_area = 6 * edge_length**2  # For a cube from edge length
    # surface_area = 6 * volume**(2/3)  # For a cube from volume
    # surface_area = (2*edge_length**2) + (4*edge_length)  # For water with a depth of only 1 meter
    emissivity = 1.0  # Not the actual value for water but we're pretending it's a perfect black-body. For actual values see: https://www.engineeringtoolbox.com/radiation-heat-emissivity-d_432.html
    # emissivity = 0.955
    water_mass = 1000 * volume  # water's density is 1000 kg/m^3
    water_heat_cap = 4184  # Water's heat capacity is 4184 Joules/kg/K
    water_albedo = 0.06

    sim_steps = 10  # Max number of steps. Will break early if equilibrium is reached.
    time_step = 3600  # W is J per second. This is the number of seconds for each step.
    time_units = None
    if time_step == 1:
        time_units = "seconds"
    elif time_step == 60:
        time_units = "minutes"
    elif time_step == 3600:
        time_units = "hours"
    elif time_step == 86400:
        time_units = "days"
    elif time_step == 604800:
        time_units = "weeks"
    else:
        print("INVALID TIME STEP")
        sys.exit(0)

    # NOTE: There's an artifact if the starting temperature is above the eventual equilibrium temperature.
    # The temperature might actually spike upwards before then descending down to equilibrium,
    # or it might spike, then drop below equilibrium before rising to equilibrium.
    # This is due to multiplying the time_step in the new amount of TSI to add to the joules.
    # It causes a large sum of joules to be added initially before new_emit is calculated and
    # subtracted in the subsequent step(s) if the step is large.
    # Also, a large time_step can cause some of the variables to overflow because numbers get too big
    # with both multiplication and division.
    # There has to be a better way. IRL there are no time steps, it's continuous and always 'seconds' but
    # the time_step causes big additions all at once in discreet 'chunks' that are not realistic; it's like
    # suddenly dumping a huge amount of joules in at once, which massively increases the temperature, which
    # immediately radiates away a large amount of energy in the next step which is a big giant fluctuation.
    old_temp = 1.0  # I'm scared to start at 0 kelvin so we'll start at 1
    old_joules = water_mass * water_heat_cap * old_temp# * time_step
    old_emit = SBC * old_temp**4 * surface_area * emissivity# * time_step
    print(" Water Temp is:", old_temp)
    print(" Starting joules:", old_joules)
    print(" Starting emit:", old_emit)

    # Also we're ignoring that water becomes ice below 273 K, and ice has a different heat capacity, albedo, etc.
    for i in range(sim_steps):  # Could do a while True
        old_joules = old_joules - old_emit + (flux * (1-water_albedo) * edge_length**2 * time_step)
        # print(" joules is:", old_joules)

        # From this page: https://www.e-education.psu.edu/earth103/node/1005
        # joules = mass * heat capacity * temperature
        # Therefore we can reverse engineer the temperature from joules / (mass * heat capacity)
        new_temp = old_joules / (water_mass * water_heat_cap)
        # print(" New temp:", new_temp)
        new_emit = SBC * new_temp**4 * surface_area * emissivity * time_step
        # print(" New emit:", new_emit)

        if round(old_temp, 8) == round(new_temp, 8):  # This is not the best method. i.e. large masses of material will change temp very slowly so this rounding will break early right at the start because the temp barely changed.
        # if old_temp == new_temp:
            print(f" Equilibrium ~= {new_temp}")
            break

        old_temp = new_temp
        old_emit = new_emit
        # if i % 24 == 0:
        print(f" Water Temp is: {new_temp:09.5f} at time step {i+1:.0f} {time_units}.")

        # Can we simply calculate the equillibrium directly?
        # water_mass * water_heat_cap * temperature = SBC * temperature**4 * surface_area * emissivity  # NOTE: This whole section is probably all kinds of wrong.
        # water_mass * water_heat_cap * temperature = SBC * (temperature*temperature*temperature*temperature) * surface_area * emissivity
        # water_mass * water_heat_cap = SBC * (temperature*temperature*temperature) * surface_area * emissivity
        # (temperature*temperature*temperature) = (SBC * surface_area * emissivity) / (water_mass * water_heat_cap)
        # temperature**3 = (SBC * surface_area * emissivity) / (water_mass * water_heat_cap)
        # temperature = ((SBC * surface_area * emissivity) / (water_mass * water_heat_cap))**(1/3)


# NOTE: Initial method for assigning temperature to whole planet using only cosine of latitude and altitude (and tilt).
@njit(cache=True, parallel=True, nogil=True)
def assign_surface_temp(verts, altitudes, radius, tilt):
    """Assign starting surface temperatures based on equatorial distance and altitude."""
    # TODO: Temps will only ever be in a range from approximately -60.0 to +60.0 C, so, what's a good dtype for that?  The world record low is approx -90 C, and the record high is approx +56 C.
    #       float16 could hold that but with little precision. float32 is probably the smallest safe choice.  I also may want room for higher and lower values than Earth produces.
    #       Consider also that Kelvin and Celsius have the same unit size; an increase of 1 K is identical to an increase of 1 C; the only difference is that C is offset by +273.15 from Kelvin values. (water freezes at 273.15 K and boils at 373.15 K)
    #       So given that I might want to someday expand the planet generator to non-Earthlike planets, or that I might need temperatures above/below Earth normals, maybe I should store them and calculate them in Kelvin.
    #       Also the fact that black body radiation is probably always done with kelvin and is tied to temperature and incoming/outgoing energy.
    surface_temps = np.zeros(len(verts), dtype=np.float32)

    # TODO: Okay, so, temp fallof with altitude is tied to humidity; more humidity = slower falloff.
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
        lat, lon = xyz2latlon(verts[v][0], verts[v][1], verts[v][2], radius)
        dist_from_equator = lat - tilt

        # Linear dist_from_equator temp: 0 at poles, 1 at equator
        # surface_temps[v] = 1 - np.abs(dist_from_equator) / 90

        # Simple modification of temp based on altitude
#        surface_temps[v] = lerp(1 - np.abs(lat) / 90, 0, altitudes[v][0])
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

        # TODO: Test splitting this into a variable for the equator falloff, square the variable, then set st[v] to that * 1-h2 because I want to see if that might run faster than doing it all on one line.
        # Something similar was true for the sample_noise function. It was faster to fill the array with a simple operation and then do multiplication at the end. Although I'm not sure if I have any simple constants to multiply against here.

    # return rescale(surface_temps, -60, 60)
    return surface_temps


# Insolation at the top of the atmosphere
@njit(cache=True, parallel=True, nogil=True)
def sample_insolation(arr, verts, radius, rotation, tilt):  # TODO: Test and see if modifying 'arr' in place is more performant than making an empty array and RETURNING it to the parent, where it will then be added to the insolation array.
    """Sample insolation for given points at a single second in time."""  # Most likely modifying in place is more performant, but by how much?
    for v in prange(len(verts)):
        x, y, z = verts[v][0], verts[v][1], verts[v][2]

        # https://www.mathworks.com/matlabcentral/answers/123763-how-to-rotate-entire-3d-data-with-x-y-z-values-along-a-particular-axis-say-x-axis
        # Rotate the planet
        rx = x*np.cos(rotation * np.pi/180) - y*np.sin(rotation * np.pi/180)
        ry = x*np.sin(rotation * np.pi/180) + y*np.cos(rotation * np.pi/180)
        rz = z

        # For a tidally locked planet we simply would not do more than 1 rotation step in the parent function that calls this function.
        # (Do not continuously increment the rotation, simply send the same rotation and tilt value every time; don't update calc_seasonal_tilt as the planet orbits either, just the initial number.)
        # (We could technically do 0 rotation steps but then the zenith would always be at lon 0 and it would be nicer
        # to allow the user to give an offset so they can determine which part of the planet experiences the solar zenith.)
        # It should be noted that in the case of tidal locking there is no "day" to accumulate or average the temperatures over; just continuous equilibrium balancing, always.

        # Tilt the planet
        tx = rx*np.cos(tilt * np.pi/180) + rz*np.sin(tilt * np.pi/180)
        ty = ry
        tz = rz*np.cos(tilt * np.pi/180) - rx*np.sin(tilt * np.pi/180)

        lat, lon = xyz2latlon(tx, ty, tz, radius)

        # dist_from_equator = lat - tilt  # Not actually using this at the moment (or probably ever)

        arr[v] += max(np.cos(np.abs(lat) * np.pi/180), 0) * max(np.cos(lon * np.pi/180), 0) # Okay, so next step is to turn this into temperatures, and then after that subtract the land temps based on altitude.
        # NOTE: Also need to compare the distribution to the simple model in the first assign_surface_temps and see if it visually looks like the same distribution. (the numbers on the scale bar will be different, of course)
    # return max(np.cos(np.abs(lat) * np.pi/180), 0) * max(np.cos(lon * np.pi/180), 0)

# Incredibly, if you divide the returned results by the number of rotation steps it appears to match the weighted cosine values from this NASA program?
# https://data.giss.nasa.gov/modelE/ar5plots/srlocat.html
# The insolation is off by some 20-something watts if you multiply that cosine times the solar constant, though...
# Insolation at the top of the atmosphere
def brute_daily_insolation(verts, altitudes, radius, tilt, snapshot=False):
    """Calculate full planet insolation for a full day for every vertex of the planet (brute force)."""
    surface_temps = np.zeros(len(verts), dtype=np.float32)
    # alt_intensity = 0  # Controls strength of altitude contribution to temp.
    # h2 = rescale(altitudes, 0, alt_intensity)

    # tilt = -90
    rotation = -180.0
    # rot_steps = 86400  # Good god this takes a long time.  A faster approximation would be to only calculate a single half-circle of points from pole to pole at fixed latitudes
    rot_steps = 360  # then do a lookup for each vert's latitude and interpolate the result.
    rot_amt = 360.0 / rot_steps

    # Add a 'locked=None' argument to the function then replace the above code with the below commented block when implementing tidal locking:
    # locked is a float that the user can specify (in combination with the tilt) to control where the solar zenith appears on the tidally locked planet.
    # if locked:
    #     rotation = locked
    #     rot_amt = 0
    # else:
    #     rotation = -180
    #     rot_amt = 360.0 / rot_steps

    for i in range(rot_steps):
        sample_insolation(surface_temps, verts, radius, rotation, tilt)
        rotation += rot_amt

        if snapshot:  #TODO: This is quite slow.
            dictionary = {}
            rescaled_i = rescale(surface_temps, 0, 255)  #NOTE: Due to the relative nature of rescale, if the min or max insolation changes then the scale will be messed up.
            dictionary[f"{i+1:03d}"] = [rescaled_i, 'gray']  # Could instead possibly just multiply the absolute value x10 or square it or something and use a 16-bit image instead of 8-bit.
            # dictionary[f"{i+1:03d}"] = [ (surface_temps + 32768).astype('uint16'), 'gray']

            pixel_data = build_image_data(dictionary)
            save_image(pixel_data, cfg.SNAP_DIR, cfg.WORLD_CONFIG["world_name"] + "_insolation_snapshot")

    # tmin = np.amin(surface_temps)
    # tmax = np.amax(surface_temps)
    # print("  Flux min is ", tmin)
    # print("  Flux max is ", tmax)

    return surface_temps  # TODO: This should be renamed to something else like insolation since this function is not working with temperature.
    # return surface_temps / 360

# Insolation at the top of the atmosphere
def calc_instant_insolation(verts, altitudes, radius, rotation, tilt):
    """Calculate full planet insolation for only 1 second given a time of day specified by rotation \
       for every vertex of the planet (brute force)."""
    insolation = np.zeros(len(verts), dtype=np.float32)
    # alt_intensity = 0  # Controls strength of altitude contribution to temp.
    # h2 = rescale(altitudes, 0, alt_intensity)

    sample_insolation(insolation, verts, radius, rotation, tilt)

    return insolation

# Insolation at the top of the atmosphere
@njit(cache=True, parallel=True, nogil=True)
def calc_insolation_slice(radius, tilt):
    """Make a slice of average (full day) insolation at one point per integer latitude. \
       I.E. sample one point from 90 to -90 degrees latitude along a single longitude."""
    points = np.zeros((181, 3), dtype=np.float64)
    result = np.zeros(181, dtype=np.float32)

    # Use trickery with negative indexing so that the latitude == the index
    # Therefore the array looks like [0, 1, ..., 89, 90, -90, -89, ..., -2, -1]
    # instead of continuous like [90, 89, 88, ..., 1, 0, -1, ..., -89, -90]
    for i in range(-90, 91, 1):
        points[i] = latlon2xyz(i, 0, radius)

    # To get more fine-grained than every integer lat we'd have to do this:
    # (but we'd lose the trick of each index being a latitude)
    # lat = 90
    # for i in range(361):
    #     points[i] = latlon2xyz(lat, 0, radius)
    #     lat -= 0.5

    rotation = -180.0
    # rot_steps = 86400
    rot_steps = 360
    rot_amt = 360.0 / rot_steps

    for x in range(rot_steps): # TODO: x is an unused var. We could do something about that, like the restructure of make_ll_arr, or a while loop, but it's not strictly needed.
        sample_insolation(result, points, radius, rotation, tilt)
        rotation += rot_amt

    # tmin = np.amin(result)
    # tmax = np.amax(result)
    # print("  Slice min is", tmin)
    # print("  Slice max is", tmax)
    # print(result)
    return result

# Insolation at the top of the atmosphere
def calc_daily_insolation(verts, altitudes, radius, tilt):
    """Calculate full planet insolation for a full day using a lookup table generated by the slice function."""
    daily_insolation = np.zeros(len(verts), dtype=np.float32)
    # alt_intensity = 0  # Controls strength of altitude contribution to temp.
    # h2 = rescale(altitudes, 0, alt_intensity)

    lookup_table = calc_insolation_slice(radius, tilt)
    interpolate_insolation(verts, lookup_table, daily_insolation, radius)

    return daily_insolation

# Insolation at the top of the atmosphere
@njit(cache=True, parallel=True, nogil=True)
def interpolate_insolation(verts, lookup_table, insolation, radius):
    """Interpolate insolation for all verts using a lookup table generated by the slice function."""
    for v in prange(len(verts)):
        lat, _ = xyz2latlon(verts[v][0], verts[v][1], verts[v][2], radius)
        lower = int(np.floor(lat))
        upper = int(np.ceil(lat))

        if -90 > lower or -90 > upper or lower > 90 or upper > 90:
            print("OUT OF RANGE SOMEHOW!")
            print("lower:", lower)
            print("upper:", upper)

        if lower == upper:
            insolation[v] = lookup_table[lower]
        else:
            # https://www.geeksforgeeks.org/how-to-implement-linear-interpolation-in-python
            # Given 2 coordinates (lower, upper) and the values at those coordinates (lookup_table)..
            # and given a 3rd coordinate (lat) that lies between them..
            # find the interpolated value at the 3rd coordinate.
            insolation[v] = lookup_table[lower] + (lat - lower) * ((lookup_table[upper] - lookup_table[lower]) / (upper - lower))

    # tmin = np.amin(insolation)
    # tmax = np.amax(insolation)
    # print("  Applied slice min is", tmin)
    # print("  Applied slice max is", tmax)
    # return insolation

def calc_yearly_insolation(points, height, radius, axial_tilt, snapshot=False):
    """Calculate full planet average insolation for a full year using a lookup table generated by the slice function."""
    # A "year" being 1 full revolution, measured in 1 degree increments.
    annual_insolation = np.zeros(len(points), dtype=np.float32)

    # TODO: We'll deal with the degrees-per-day later, e.g. 360 divided by days-per-year to get
    # the proper value to plug into calc_seasonal_tilt for the second number.

    # for x in range(cfg.WORLD_CONFIG["year_length"]):
    #     current_tilt = calc_seasonal_tilt(axial_tilt, day2deg(cfg.WORLD_CONFIG["year_length"], x))
    for x in range(360):
        current_tilt = calc_seasonal_tilt(axial_tilt, x)
        insolation = calc_daily_insolation(points, height, radius, current_tilt)
        annual_insolation += insolation  # Accumulate insolation for x number of days

        if snapshot:
            dictionary = {}
            rescaled_i = rescale(insolation, 0, 255)  #NOTE: Due to the relative nature of rescale, if the min or max insolation changes then the scale will be messed up.
            dictionary[f"{x+1:03d}"] = [rescaled_i, 'gray']  # Could instead possibly just multiply the absolute value x10 or square it or something and use a 16-bit image instead of 8-bit.
            # dictionary[f"{x+1:03d}"] = [ (insolation + 32768).astype('uint16'), 'gray']

            pixel_data = build_image_data(dictionary)
            # save_image(pixel_data, cfg.SNAP_DIR, "day")
            save_image(pixel_data, cfg.SNAP_DIR, cfg.WORLD_CONFIG["world_name"] + "_insolation_day")

    return annual_insolation


# Broken stuff that doesn't work right.
def calc_hour_angle_insolation(tsi):
    """Solar flux at specific latitude, time, and day of year"""

    solstice = 173
    year_length = 365.25
    day_length = 24
    half_day = day_length / 2

    hour = 12
    day = 80
    latitude = 0
    longitude = 0

    axial_tilt = 23.44
    current_tilt = calc_seasonal_tilt(axial_tilt, 90)

#        hour_angle = ( ((hour - half_day) * np.pi) / day_length ) + ((longitude * np.pi) / 180)
    # hour_angle = 90
    hour_angle = (360 / day_length) * (hour - half_day)  # In degrees

    declination = current_tilt * (np.pi/180) * np.cos( (2*np.pi * (day - solstice)) / year_length )

#        hour_angle = np.arccos(-np.tan(latitude) * np.tan(declination))# * np.pi/180

    # zenith = ( np.sin(latitude * (np.pi/180)) * np.sin(declination) + np.cos(latitude * (np.pi/180)) ) * ( np.cos(declination) * np.cos(hour_angle) )
    zenith = ( np.sin(latitude * (np.pi/180)) * np.sin(declination * (np.pi/180)) + np.cos(latitude * (np.pi/180)) ) * ( np.cos(declination * (np.pi/180)) * hour_angle )

    flux = tsi * np.cos(zenith * (np.pi/180))# * longwave_transmittance
    # flux = tsi * (1-world_albedo) * np.cos(zenith)# * longwave_transmittance

    test = tsi * np.cos(np.abs(latitude - current_tilt) * (np.pi/180)) * np.cos(hour_angle * (np.pi/180))  # Not quite right; when hour is 12 and lat = 0 it should be at full strength

    print(f"TSI is {tsi}")
    print(f"Hour angle is {hour_angle}")
    print(f"Declination angle is {declination}")
    print(f"Zenith angle is {zenith}")
    print(f"Flux at Lat: {latitude}, Lon: {longitude} at hour {hour} is: {flux}")
    print(f"Test is {test}")

# TODO: Should it be called humidity or moisture? This is the water content of the air.  Pretty sure it's humidity.
# TODO: Do some book learnin' and find out if altitude modifies humidity. And if temperature modifies humidity, too.
# https://en.wikipedia.org/wiki/Properties_of_water
def assign_surface_humidity(verts, water_mask):
    # TODO: What's the best dtype for humidity? Humidity can only range from 0-100 but I may want decimals. np.float16 might get me 1 or 2 decimal places, rounded. float32 gets me 5 or 6 decimal places, rounded. uint8 would be more than enough for int values.
    #       Once I have the climate system working in float32 it might be interesting to go back and change things out for float16 and see how much things change.
    humidity = np.zeros(len(verts), dtype=np.float32)

    return humidity

# https://blender.stackexchange.com/questions/43801/what-is-the-c-mathf-lerp-equivalent-in-python
@njit(cache=True)
def lerp(input1, input2, mask):
    """Lerp between two values based on a 0-1 mask value.
    When mask is 0, return input1. When mask is 1, return input2.
    Otherwise blend between the two."""
    return ((1 - mask) * input1) + (mask * input2)

# Nixis
A python program for generating planet-scale maps for spherical worlds.

While there are many existing tools (such as [Gaea](https://quadspinner.com/), [TerreSculptor](http://www.demenzunmedia.com/home/terresculptor/), [World Machine](https://www.world-machine.com/), and [World Creator](https://www.world-creator.com/)) that can create plausibly realistic-looking terrain for small areas (a few kilometers squared) there are few to none that do so for a full-sized spherical planet.  In this instance "plausibly realistic-looking" refers to a terrain that incorporates principles of hydraulic erosion and weathering such that it resembles real terrain that has been acted upon by these natural forces over time.  Existing tools that apply erosion algorithms on small or flat terrains are not programmed to compensate for the distortion that is introduced by mapping a sphere onto a rectangular image.

## What This Project Is
* The goal of this project is to use principles of hydraulic erosion and tectonic activity to create large-scale features like continents, mountain chain distributions, rivers, and water sheds that appear believable at planetary scales on a spherical planet. 
* Given that the objective is planetary scales, the size of the smallest feature is not expected to be smaller than about one kilometer or a few kilometers per pixel.  See NASA's Blue Marble full Earth topography and bathymetry maps for reference.
* The output of the tool is saved as a series of [equirectangular texture maps](https://en.wikipedia.org/wiki/Equirectangular_projection) that can be used in 3D graphics applications such as Maya, Blender, Unreal Engine, Unity, etc. for video game graphics, visual effects, or animation. 

## What This Project *Is Not*
* A fully scientific planetary simulation is not the goal of this project.  Shortcuts and approximations are used for the sake of simplicity and performance.  As long as the output of the program looks *plausibly* realistic it need not be perfectly realistic or fully scientific for our purposes.
* Nor is it a goal to be able to view a planet at resolutions from orbital altitudes to ground level.  This tool is not for creating playable maps for video games.  Its explicit goal is to create textures, that will wrap onto spheres for visualization, which do not have the resolution to be playable at ground level.  These textures are intended to be viewed from orbital distances. 
* Flat worlds, disc worlds, ring worlds, and terrains that wrap in both the X and Y direction need not apply.  There are many tools that can already cater to those styles.  Nixis is for spherical planets.

## Assumptions and Limitations
* Calculations assume that the planet is spherical or very close to spherical (i.e. the calculations would be skewed if you made a small potato-like moon or asteroid).
* Calculations assume the planet's distance from its star is constant (a perfectly circular orbit).
* Measurements and calculations are in metric units.
* Planets don't have moons.
* No support for binary (or more) stars or planets.
* No support for tidally locked planets.
* Nixis gobbles lots of RAM.

## Roadmap (in no particular order)
* Hydraulic erosion.
* Plate tectonics and hotspots.
* Precipitation.
* Climate and biome maps derived from temperature and precipitation.
* Elliptical planet orbits.
* Tidally locked planets.
* Vegetation growth.
* Track soil fertility.
* Distribution of different minerals and resources (e.g. limestone, sandstone, useful metals, etc.).
* Climate-driven winds and pressures and weather.

-----
## options.json
Important settings are stored in the options.json file in the main directory. While most of these should be self-explanatory a few might not be obvious so they are documented here.
```python
{
    "save_folder": "output",        # Name of the folder where exports are saved (this folder is created in the main directory)
    "snapshot_folder": "snapshots", # Name of the folder where snapshots are saved (this folder is created in the save_folder)
    "img_width": 2048,              # Width of exported images (It is recommended that this be 2x the height)
    "img_height": 1024,             # Height of exported images (It is recommended that this be 0.5x the width)
    "img_format": "png",            # Format of exported images (In theory any format supported by Pillow should work)
    "bit_depth": 16,                # Bit-depth per channel of exported images
    "export_list": {},              # A dict of which maps should be exported (e.g. height, temperature, biome, etc.)
    "mesh_format": "obj",           # Format of exported 3D mesh files
    "point_format": "ply",          # Format of exported 3D point clouds
    "settings_format": "json",      # Format of Nixis settings and exported world seeds
    "database_format": "sqlite3"    # Format of exported planet databases
}
```

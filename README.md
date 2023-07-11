<div align="center">

**⚠ Nixis is still early in development and will undergo many breaking changes. ⚠  
The structure is a mess and the files are full of enormous tracts of commented notes and it is most definitely not pep-8 compliant.**

</div>

# Nixis
A python program for procedurally generating planet-scale maps for Earth-like, spherical worlds.

![](https://github.com/MightyBOBcnc/nixis/blob/main/docs/nixis_preview.jpg)

While there are many existing tools (such as [Gaea](https://quadspinner.com/), [TerreSculptor](http://www.demenzunmedia.com/home/terresculptor/), [World Machine](https://www.world-machine.com/), and [World Creator](https://www.world-creator.com/)) that can create plausibly realistic-looking terrain for small areas (a few kilometers squared) there are few to none that do so for a full-sized spherical planet.  In this instance "plausibly realistic-looking" refers to a terrain that incorporates principles of hydraulic erosion and weathering such that it resembles real terrain that has been acted upon by these natural forces over time.  Existing tools that apply erosion algorithms on small or flat terrains are not programmed to compensate for the distortion that is introduced by mapping a sphere onto a rectangular image.

## What This Project Is
* The goal of this project is to use principles of hydraulic erosion and tectonic activity to create large-scale features like continents, mountain chain distributions, rivers, and water sheds that appear believable at planetary scales on a spherical planet. 
* Given that the objective is planetary scales, the size of the smallest feature is not expected to be smaller than about one kilometer or a few kilometers per pixel.  See NASA's Visible Earth (Blue Marble) [topography](https://visibleearth.nasa.gov/images/73934/topography) and [bathymetry](https://visibleearth.nasa.gov/images/73963/bathymetry) maps for reference.
* The output of the tool is saved as a series of [equirectangular texture maps](https://en.wikipedia.org/wiki/Equirectangular_projection) that can be used in 3D graphics applications such as Maya, Blender, Unreal Engine, Unity, etc. for video game graphics, visual effects, or animation. (Other map projections may be forthcoming at a later time that would be more useful for e.g. a D&D or other TTRPG campaign map; see roadmap below.)

## What This Project *Is Not*
* A fully scientific planetary simulation is not the goal of this project.  Shortcuts and approximations are used for the sake of simplicity and performance.  As long as the output of the program looks *plausibly* realistic it need not be perfectly realistic or fully scientific for our purposes.
* Nor is it a goal to be able to view a planet at resolutions from orbital altitudes to ground level.  This tool is not for creating playable maps for video games.  Its explicit goal is to create textures, that will wrap onto spheres for visualization, which do not have the resolution to be playable at ground level.  These textures are intended to be viewed from orbital distances. 
* Flat worlds, disc worlds, ring worlds, and terrains that wrap in both the X and Y direction need not apply.  There are many tools that can already cater to those styles.  Nixis is for spherical planets.

## Assumptions and Limitations
* **At the moment Nixis can only export grayscale maps for elevation, solar insolation, and an ocean/land mask. RGB exports will be added later.**
* Calculations assume that the planet is spherical or very close to spherical (i.e. the calculations would be skewed if you made a small potato-like moon or asteroid).
* Calculations assume the planet's distance from its star is constant (a perfectly circular orbit).
* Measurements and calculations are in metric units.
* Planets don't have moons.
* No support for binary (or more) stars or planets.
* No support for tidally locked planets.
* Nixis gobbles lots of RAM when you specify a large number of divisions with the `-d` or `--divisions` argument.

## Roadmap (in no particular order)
### Planned features:
* Temperature driven by solar insolation (insolation has been implemented but needs to be converted into temperature).
* Hydraulic erosion.
* RGB image exports for land and ocean coloration.
* Plate tectonics and hotspots.
* Precipitation.
* Climate and biome maps derived from temperature and precipitation.
* Import an existing height map and run erosion/climate/biome calculations on it.
* Vegetation growth.
* Track soil fertility.
* Distribution of different minerals and resources (e.g. limestone, sandstone, useful metals, etc.).
### Tentative features (subject to cancellation):
* Some additional map projections for export (e.g. [Mercator](https://en.wikipedia.org/wiki/Mercator_projection)).
* .svg export of river networks and watersheds (and possibly other geological features).
* Insolation and climate-driven winds and pressures and weather.
* Elliptical planet orbits.
* Tidally locked planets.
* Clouds.

-----
## Installation and Use
(Better instructions coming later.)

Nixis is being developed in an [Anaconda environment](https://www.anaconda.com/) with Python >3.8.x (older versions have not been tested against but 3.10.x and 3.11.x are known to work). You can install the environment from `requirements.yaml`.

After setting up your environment you can run `python nixis.py -h` or `python nixis.py --help` for arguments. 

A smart workflow would be to generate a few planets with random seed numbers and a default number of divisions without exporting maps to quickly find a seed that you like, then increase the resolution and divisions with your chosen seed for a more detailed export. Higher levels of subdivision and larger images take longer to process.

Erosion and solar insolation are both still in development so you'll have to set some flags in `nixis.py` manually (after the section that sets up argparse): 
```
do_erode = False
do_climate = False
```
Then tinker with the erosion and insolation references further down in `nixis.py` which you can find by searching for `if do_erode:` or `if do_climate:`, and optionally changing the color gradient used for visualization near the bottom of `nixis.py` by setting this line: 
```python
scalars = {"s-mode":"elevation", "scalars":height}
```
where `s-mode` choses which gradient to use and `scalars` defines which values to apply the gradient to.  See `def make_scalars` in `gui.py` for options (this is also a work in progress).  E.G. you might choose `"insolation"` as the gradient and `daily_insolation` as the scalars if do_climate is True, or `"temperature"` and `surface_temps`.

NOTE: Erosion modifies the existing `height` array in place so setting the scalars to height will work regardless of if do_erode is False but setting the scalars to any of the temperature or insolation arrays will error if do_climate is False as they won't exist.

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
    "bit_depth": 16,                # (Currently unused) Bit-depth per channel of exported images
    "export_list": {},              # (Currently unused) A dict of which maps should be exported (e.g. height, temperature, biome, etc.)
    "mesh_format": "obj",           # Format of exported 3D mesh files
    "point_format": "ply",          # (Currently unused) Format of exported 3D point clouds
    "settings_format": "json",      # Format of exported world seeds
    "database_format": "sqlite3"    # (Currently unused) Format of exported planet databases
}
```

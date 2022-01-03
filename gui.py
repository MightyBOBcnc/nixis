"""Functions for rendering the final result in PyVista."""

import time
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from util import latlon2xyz, find_percent_val, find_val_percent

import cmocean
# zmaps = cmocean.tools.get_dict(cmocean.cm.thermal, N=9)
# zmaps = cmocean.cm.thermal
# zmaps = cmocean.cm.get_cmap("thermal").copy()

def add_points(pl, points):
    # Is it strictly necessary that these be np.arrays?
    for key, data in points.items():
        dots = pv.PolyData(np.array(data))
        pl.add_mesh(dots, point_size=10.0, color=key)


def add_lines(pl, radius, tilt):
    x_axisline = pv.Line([-1.5*radius,0,0],[1.5*radius,0,0])
    y_axisline = pv.Line([0,-1.5*radius,0],[0,1.5*radius,0])
    z_axisline = pv.Line([0,0,-1.5*radius],[0,0,1.5*radius])

    # Axial tilt line
    # ax, ay, az = latlon2xyz(tilt, 45, radius)
    ax, ay, az = latlon2xyz(tilt, 0, radius)
    t_axisline = pv.Line([0,0,0], [ax * 1.5, ay * 1.5, az * 1.5])

    # Sun tilt line (line that is perpendicular to the incoming solar flux)
    # ax, ay, az = latlon2xyz(90-tilt, -135, radius)
    ax, ay, az = latlon2xyz(90-tilt, 180, radius)
    s_axisline = pv.Line([0,0,0], [ax * 1.5, ay * 1.5, az * 1.5])

    pl.add_mesh(x_axisline, line_width=5, color = "red")
    pl.add_mesh(y_axisline, line_width=5, color = "green")
    pl.add_mesh(z_axisline, line_width=5, color = "blue")
    pl.add_mesh(t_axisline, line_width=5, color = "magenta")
    pl.add_mesh(s_axisline, line_width=5, color = "yellow")


def visualize(verts, tris, heights=None, scalars=None, zero_level=0.0, surf_points=None, radius=1.0, tilt=0.0):
    """Visualize the output."""
    pl = pv.Plotter()


    if scalars["s-mode"] in ("insolation", "temperature"):
        show_ocean_shell = False
    else:
        show_ocean_shell = True

    # pyvista expects that faces have a leading number telling it how many
    # vertices a face has, e.g. [3, 0, 11, 5] where 3 means triangle.
    # https://docs.pyvista.org/examples/00-load/create-poly.html
    # So we fill an array with the number '3' and merge it with the cells
    # from meshzoo to get a 'proper' array for pyvista.
    time_start = time.perf_counter()
    tri_size = np.full((len(tris), 1), 3)
    new_tris = np.hstack((tri_size, tris))
    time_end = time.perf_counter()
    print(f"  Time to reshape triangle array: {time_end - time_start :.5f} sec")

    tri_size = None

    # Create pyvista mesh from our icosphere
    time_start = time.perf_counter()
    planet_mesh = pv.PolyData(verts, new_tris)
    time_end = time.perf_counter()
    print(f"  Time to create the PyVista planet polydata: {time_end - time_start :.5f} sec")

    # Separate mesh for ocean water
    if show_ocean_shell:
        ocean_shell = pv.ParametricEllipsoid(radius, radius, radius, u_res=300, v_res=300)
        pl.add_mesh(ocean_shell, show_edges=False, smooth_shading=True, color="blue", opacity=0.15)

    # Add any surface marker points or other floating marker points
    if surf_points is not None:
        add_points(pl, surf_points)

    # Add axis lines
    add_lines(pl, radius, tilt)

    # minval = np.amin(scalars["scalars"])
    # maxval = np.amax(scalars["scalars"])
    # heights = np.clip(heights, minval*1.001, maxval)

    # Prepare scalar gradient, scalar bar, and annotations
    color_map, anno = make_scalars(scalars["s-mode"], scalars["scalars"])
    sargs = dict(n_labels=0, label_font_size=12, position_y=0.07)

    # ToDo: Add title to the scalar bar sargs and dynamically change it based on what is being visualized (e.g. Elevation, Surface Temperature, etc.)
    # title="whatever" (remove the quotes and make 'whatever' into a variable, like the s-mode or whatever. like title=scalars["s-mode"])
    # "Current"? ".items()"? https://stackoverflow.com/questions/3545331/how-can-i-get-dictionary-key-as-variable-directly-in-python-not-by-searching-fr
    # https://stackoverflow.com/questions/16819222/how-to-return-dictionary-keys-as-a-list-in-python

    time_start = time.perf_counter()
    # pl.add_mesh(planet_mesh, show_edges=False, smooth_shading=True, color="white", below_color="blue", culling="back", scalars=scalars["scalars"], cmap=custom_cmap, scalar_bar_args=sargs, annotations=anno)
    pl.add_mesh(planet_mesh, show_edges=False, smooth_shading=True, color="white", culling="back", scalars=scalars["scalars"], cmap=color_map, scalar_bar_args=sargs, annotations=anno)
    time_end = time.perf_counter()
    print(f"  Time to add the PyVista planet mesh to the plotter: {time_end - time_start :.5f} sec")

    pl.show_axes()
    pl.enable_terrain_style(mouse_wheel_zooms=True)  # Use turntable style navigation
    print("Sending to PyVista...")
    pl.show()


def make_scalars(mode, scalars):
    # Define the colors we want to use (NOT BEING USED AT THE MOMENT)
    # blue = np.array([12/256, 238/256, 246/256, 1])
    # black = np.array([11/256, 11/256, 11/256, 1])
    # grey = np.array([189/256, 189/256, 189/256, 1])
    # yellow = np.array([255/256, 247/256, 0/256, 1])
    # red = np.array([1, 0, 0, 1])

    minval = np.amin(scalars)
    maxval = np.amax(scalars)

    if mode == "topography":
        pass
    elif mode == "bathymetry":
        color_map = "deep_r"
        anno = {minval:f"{minval:.2}", maxval:f"{maxval:.2}"}
    elif mode == "ocean":
        pass
    elif mode == "insolation":
        color_map = "solar"
        anno = {minval:f"{minval:.2}", maxval:f"{maxval:.2}"}
    elif mode == "temperature":
        color_map = "thermal"
        anno = {minval:f"{minval:.2}", maxval:f"{maxval:.2}"}
    elif mode == "elevation":
        # Derive percentage of transition from ocean to land from zero level
        # shore = ( (zero_level - minval) / (maxval - minval) ) - 0.001
        shore = (find_val_percent(minval, maxval, 0) / 100) - 0.001
        nearshore = shore - 0.001
        print("cmap transition lower:", nearshore)
        print("cmap transition upper:", shore)

        color_map = LinearSegmentedColormap.from_list('ocean_and_topo', [(0, [0.1,0.2,0.6]), (nearshore, [0.8,0.8,0.65]), (shore, [0.3,0.4,0.0]), (1, [1,1,1])])

        # https://matplotlib.org/cmocean/
        # https://docs.pyvista.org/examples/02-plot/cmap.html
        # https://colorcet.holoviz.org/
        # sargs = dict(below_label="Ocean", n_labels=0, label_font_size=15)
        # anno = {minval:f"{minval:.2}", zero_level:"0.00", maxval:f"{maxval:.2}"}
        anno = {minval:f"{minval:.2}", find_percent_val(minval, maxval, nearshore*100):"0.00", maxval:f"{maxval:.2}"}
    else:
        print("NO VALID SCALAR MODE SPECIFIED")

    return color_map, anno

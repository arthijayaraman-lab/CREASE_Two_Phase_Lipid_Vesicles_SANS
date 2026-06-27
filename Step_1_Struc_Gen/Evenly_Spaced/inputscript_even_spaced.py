###########################################################################################################################

# Code created by Rohan S. Adhikari, Postdoctoral Researcher, Jayaraman Lab, University of Delaware, Newark, DE.
# Code created in March 2026.
# This code takes as input the list of phase grids generated using a constrained random walk in spherical coordinates ...
# ... and distributes the isolated phase domains evenly along the two-phase grid.
# This code can be run using the command python/python3 inputscript_even_spaced.py
# This code uses the module evenly_spaced_domains.py for spreading out the isolated domains.
# This code uses the file paper.mplstyle for setting the fonts and style of the output flattened representation image.

# Code input files - Grid list output by the constrained random walk algorithm (such as 'Liquid_Grids_Sample.txt').
# Code output files - (1) Grid list containing phase grids of evenly spaced domains. ...
# ... (2) Point scatterer representation of the vesicles with evenly spaced domains ...
# ... (3) Flattened representation of phase grids with the evenly spaced domains.

# Happy computing! May your jobs forever run instantly on the computational cluster!

############################################################################################################################

import numpy as np
from evenly_spaced_domains import evenly_spaced_domains

params = {

    "num_grids": 1024, # Total number of grids over which the random walk will be performed (1024 for 32*32 grids), must be a square number.
    "lvf": 0.20, # Liquid phase volume fraction 
    "num_liq_dom": 10, # Number of isolated liquid domains
    "vesc_core_rad": 214, # Core radius of the two phase vesicle (for point scatterers representation)
    "vesc_shell_thick": 36, # Shell thickness of the two phase vesilce (for point scatterers representation)
    "num_point_scatterers": 1024, # Number of point scatterers to fill the shell of the lipid vesicle
    "boxlength": 700 # Boxlength for packing the lipid vesicle (set slightly larger than total diameter of vesicle). Only useful for visualizing point scatterers rep. on OVITO.

}

file_names = {

    "input_grid_file": 'Liquid_Grids_Example.txt',
    "out_ps_dump_file": 'Point_Scatterers_Evenly_Spaced.dump', # Point Scatterers representation of two phase vesicle. Can be visualized using OVITO.
    "out_grids_list": 'Evenly_Spaced_Grids_Example.txt', # List of Liquid Phase Grids for further calculations.
    "out_flattened_rep": 'Evenly_Spaced_Flattened_Representation_Example.png' # A flattened representation of the vesicle in cartesian coordinates for ease of visualization.

}

evenly_spaced_domains(params, file_names)

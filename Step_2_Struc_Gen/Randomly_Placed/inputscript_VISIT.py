###########################################################################################################################

# Code created by Rohan S. Adhikari, Postdoctoral Researcher, Jayaraman Lab, University of Delaware, Newark, DE
# Code created around December 2025
# This code is an input script to the VISIT (vesicles with irregularly shaped isolated-domains in two-phases) algorithm. 
# The phase represented using a random walk can be partitioned into user specified number of isolated domains.
# The user also specifies the phase fraction for the phase that they wish to represent using random walk.
# The isolated phase domains are a collection of phase grids corresponding to one phase ...
# ... that are surrounded by the other phase in every grid.
# This code places the isolated domains at random locations on the two phase grid.
# The code iterates until it finds the configuration required by the user. There is a theoretical maximum number of ...
# ... isolated domains that can be obtained which is a function of the phase volume fraction and ...
# ... the total number of discretized grids. 
# If you choose a high volume fraction and high number of isolated domains the code might keep iterating indefinitely. 
# Always start with lower number of domains and increase gradually while being mindful of your available resources 

# To run this code, specify the required parameters and files names and run using python inputscript_crwsg.py / ...
# ... python3 inputscript_crwsg.py.
# This code uses the constr_rand_walk_sph_grids.py file to perform the user specified constrained random walk ...
# ... in spherical grids. But do not run using python constr_rand_walk_sph_grids.py, since that is just a module being ...
# ... referenced here.
# This code uses paper.mplstyle file to set the fonts and plot style for the flattened vesicle representation.
# This code used the following Python modules, numpy, random, scipy, matplotlib.pyplot, and time.
# Make sure you have installed the necessary modules before running this script.
# Don't forget to have loads of fun computing! :)   

###########################################################################################################################

# Load the necessary modules.

import numpy as np
from VISIT import VISIT

# The necessary parameters for constraining the random walk
# Only change the values on the right side of the colon, not the variable names to the left which are used within the 
# ... constr_rand_walk_sph_grids.py module.
# If you do not need a point scatterers representation just set the num_point_scatterers to 0.
# Boxlength is only useful if you are planning to visualize the point scatterers representation on OVITO.
# If yes, then set boxlength to slightly larger than the vesicle diameter.  

params = {

    "num_grids": 1024, # Total number of grids over which the random walk will be performed (1024 for 32*32 grids) 
    # num_grids must be a square number.
    "lvf": 0.20, # Liquid phase volume fraction 
    "num_liq_dom": 10, # Number of isolated liquid domains
    "vesc_core_rad": 214, # Core radius of the two phase vesicle (for point scatterers representation)
    "vesc_shell_thick": 36, # Shell thickness of the two phase vesilce (for point scatterers representation)
    "num_point_scatterers": 1024000, # Number of point scatterers to fill the shell of the lipid vesicle
    "boxlength": 700 # Boxlength for packing the lipid vesicle (set slightly larger than total diameter of vesicle). 
    # boxlength only useful for visualizing point scatterers rep. on OVITO.

}

# Specify the file names (and directory) for the output files.
# If you choose to output the files to different directory, make sure that the directory exists.
# The point scatterer representation (.dump file) can be visualized using OVITO.
# Flattened representation is best for visualizng the isolated domains and the random walk. 
# Remember that the flattened representation is periodic in phi and not in cos(theta), ... 
# ... because a sphere goes back to the starting point in the azimuthal direction and not the polar direction.


file_names = {

    "out_ps_dump_file": 'Point_Scatterers_Two_Phase_Vesicle.dump', # Point Scatterers representation of two phase vesicle. 
    # .dump files can be visualized using OVITO.
    "out_grids_list": 'Liquid_Grids_Example.txt', # List of Liquid Phase Grids for further modifications.
    # Grid list is useful for modifications such as spacing out domains or clustering them
    "out_flattened_rep": 'Flattened_Representation_Example.png' 
    # A flattened representation of the vesicle in cartesian coordinates for ease of visualization.

}

VISIT(params, file_names)

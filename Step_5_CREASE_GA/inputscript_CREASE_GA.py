###########################################################################################################################

# Code created by Rohan S. Adhikari, Postdoctoral Researcher, Jayaraman Lab, University of Delaware, Newark, DE.
# Code created in April 2026.
# This is an input script for CREASE-GA optimization of dDPPC-DLPC lipid vesicles SANS data.
# Code outputs distribution of structural features whose computed scattering profile closely matches input experimental ...
# ... profile.
# Change the values for parameters and set file names here and run using python/python3 inputscript_CREASE_GA.py.

# List of input files: (1) Experimental SANS data containing 4 columns and commented out text if any. The 4 columns of ...
# the SANS data should correspond to q [Angstrom^-1], I(q) [a.u. or cm^-1], error in I(q) [a.u. or cm^-1], ... 
# ... and errors in q [Angstrom^-1], respectively.
# List of input files: (2) Trained XGBoost ML model that links a set of structural features directly to its computed I(q).

# List of output files: (1) GA table (.csv) that contains all candidate solutions from last generation of CREASE-GA ... 
# ... along with X2 values. Each candidate solution is a set of structural feature values. 4 for dDPPC-DLPC lipid vesicles.
# List of output files: (2) Best candidate (lowest X2) I(q) from the last generation of all CREASE-GA runs, along with ...
# ... standard deviation in best candidate I(q) from the last generation of every CREASE-GA run.

# This script incorporates resolution smearing within CREASE-GA optimization. User can set the required number of points ...
# ... for resolution smearing.  

# EnGA and Enjoy!

############################################################################################################################

import numpy as np
from multiple_runs_res_smear_CREASE_engine import multiple_runs_res_smear_CREASE_engine

struc_feat_range = {

    "rcore_low": 170, # Vesicle core radius lower range in Angstroms.
    "rcore_high": 330, # Vesicle core radius higher range in Angstroms.
    "lvf_low": 0.135, # Liquid phase volume fraction lower range.
    "lvf_high": 0.139, # Liquid phase volume fraction higher range.
    "ndom_low": 1, # Lower range for number of isolated liquid domains.
    "ndom_high": 50, # Higher range for number of isolated liquid domains.
    "sld_low": -4.0, # Lower range of liquid to gel SLD contrast ratio
    "sld_high": -3.0 # Higher range of liquid to gel SLD contrast ratio.

}

GA_res_params = {

    "ini_popsize": 1200, # Number of individuals in the CREASE-GA initial population.
    "popsize": 100, # Number of individuals in the CREASE-GA population.
    "num_gen": 200, # Number of CREASE-GA generations. 
    "num_ga_runs": 5, # Number of CREASE-GA independent trials.
    "num_res_points": 11 # Number of points for resolution smearing of computed profiles.

}

input_files = {

    "exp_data": './Exp_Data/q_Iq_Err_2_9C.txt', # Experimental SANS data.
    "xgb_model": './xgb_models/xgbmodel_vesicles_evenly_spaced.json' # Trained XGBoost surrogate ML model.

}

outfile_names = {

    "struc_feat_table": './ga_table_25_runs_9C.csv', # A csv file that contains all last generation CREASE candidate solutions across multiple runs.
    "best_x2_iq": './Iq_CREASE_Candidate_Lowest_X2.txt' # A .txt file that contains I(q) of lowest X2 candidate across multiple runs and error bars in I(q)..

}

multiple_runs_res_smear_CREASE_engine(struc_feat_range, GA_res_params, input_files, outfile_names)

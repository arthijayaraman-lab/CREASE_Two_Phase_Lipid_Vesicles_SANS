#########################################################################################################################################

# Code modified by Rohan S. Adhikari, Postdoctoral Researcher, Jayaraman Lab, University of Delaware, Newark, DE.
# Code modified in April 2026. 
# This is a module for the CREASE-GA optimization loop which accounts for resolution smearing and allows for multiple GA runs.
# This module is called within 'inputscript_CREASE_GA.py'. Do not run this module on its own.
# Change values for parameters and set file names in 'inputscript_CREASE_GA.py' and run using python/python3 inputscript_CREASE_GA.py 

########################################################################################################################################

import numpy as np
import sys
import xgboost as xgb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import warnings
import pandas as pd
import time
from scipy.special import i0, i0e
warnings.filterwarnings('ignore')

def multiple_runs_res_smear_CREASE_engine(struc_feat_range, GA_res_params, input_files, outfile_names):

    ini_time = time.time()

    rcore_min = struc_feat_range['rcore_low']
    rcore_max = struc_feat_range['rcore_high']

    lvf_min = struc_feat_range['lvf_low']
    lvf_max = struc_feat_range['lvf_high']

    ndom_min = struc_feat_range['ndom_low']
    ndom_max = struc_feat_range['ndom_high']

    sldl_min = struc_feat_range['sld_low']
    sldl_max = struc_feat_range['sld_high']

    def calculate_fitness(data1, data2, data3, f, c):
        """
    Calculate fitness scorecbetween any two matrices using Structural Similarity Index (SSIM).

    Parameters:
    - data1 (np.ndarray): First matrix.
    - data2 (np.ndarray): Second matrix.

    Returns:
    - float or None: The SSIM score if successful, else None.
    """

        data2                                = data2/data2[0]
        data2                                = data2*f + c

        log_error = 0.0	
        
        for i in range(numq):
            log_error = log_error + (data2[i] - data1[i])**2.0/data3[i]**2.0  

        log_error = log_error/numq

        return log_error

    def create_combined_data(struc_features):
        shape_struc_features=struc_features.shape
        shape_q = q_res.shape
        repeated_struc_features = np.repeat(struc_features, repeats=shape_q, axis=0)
        repeated_q = (np.tile(np.log10(q_res), shape_struc_features[0])).reshape(-1, 1)
        combined_data = np.hstack((repeated_struc_features,repeated_q))
        
        return combined_data
        
    def genes_to_struc_features(genevalues):
        #Volume fraction 
        rcore_mu = genevalues[:,0]*(rcore_max - rcore_min) + rcore_min
        lvf      = genevalues[:,1]*(lvf_max - lvf_min) + lvf_min   
        ndom_mu  = genevalues[:, 2]*(ndom_max - ndom_min) + ndom_min
        ndom_mu_int = [int(item) for item in ndom_mu]
        sld_vals = genevalues[:, 3]*(sldl_max - sldl_min) + sldl_min
        struc_features = np.vstack((rcore_mu, lvf, ndom_mu_int, sld_vals))
        struc_features=struc_features.transpose()
        return struc_features

    def generate_profile(combined_data):
        feature_names = ["Rcore", "LVF", "Ndom", "SLDL", "q"]
        dmatrix = xgb.DMatrix(combined_data, feature_names=feature_names)
        generated_profile = xgboost_model.predict(dmatrix)
        return generated_profile

    def generateallprofiles(gatable):
        """
        Generate profiles for all individuals in the genetic table .

        Parameters:
            - gatable (np.ndarray): Genetic table containing gene values.
            - profile_generator (ProfileGenerator): An instance of the ProfileGenerator class.

        Returns:
            - np.ndarray: Generated profiles for all individuals.
        """
        popsize=gatable.shape[0]
        indscore=gatable.shape[1]-1
        strucfeatures = genes_to_struc_features(gatable[:,0:indscore])
        combined_data = create_combined_data(strucfeatures)
        shape_combined_data=combined_data.shape
        generated_profiles=np.zeros((numq,popsize))
        for n in range(popsize):
            inputdata=combined_data[int(n*numq_res):int((n+1)*numq_res),:]
            generated_profile=generate_profile(inputdata)
            res_smeared_profile = np.zeros(numq)

            for qi in range(numq):
                for res_point in range(1, num_res_points):
                    crp = int(res_point + qi*num_res_points)
                    crpp = int(crp - 1)
                    res_smeared_profile[qi] += ((10**generated_profile[crp])*res_func[crp] + (10**generated_profile[crpp])*res_func[crpp])*(q_res[crp] - q_res[crpp])/2

            generated_profiles[:,n]=res_smeared_profile
        return generated_profiles

    def updateallfitnesses(gatable,profiles):
        """
        Update fitness scores for all individuals in the genetic table.

        Parameters:
            - gatable (np.ndarray): Genetic table containing gene values and fitness scores.
            - profiles (np.ndarray): Generated profiles for all individuals.
            - inputdata (np.ndarray): Input data for fitness calculation.
            
        Returns:
            - np.ndarray: Updated genetic table with fitness scores.
        """
        popsize=gatable.shape[0]
        indscore=gatable.shape[1]-1
        for n in range(popsize):
            gatable[n,indscore]=calculate_fitness(input_data, profiles[:,n], err_data, gatable[n,4]*(f_range_high - f_range_low) + f_range_low, gatable[n,5]*(c_range_high - c_range_low) + c_range_low)
        return gatable

    def generate_children(parents):
        """
        Generate children by crossover from parent individuals.

        Parameters:
            - parents (np.ndarray): Parent individuals.

        Returns:
            - np.ndarray or None: Children generated by crossover.
        """
        size_parents = parents.shape
        numparents = size_parents[0]
        numchildren = popsize - numparents
        if numchildren % 2 !=0:
            print('numchildren must be even!')
            return None
        numpairs = int(numchildren/2)
        numcols = size_parents[1]
        #Using rank weighting for parent selection
        randnumbersparent = np.random.rand(numchildren)
        #each two consecutive rows mate
        parentindices = np.int64(np.floor((2*numparents+1-np.sqrt(4*numparents*(1+numparents)*(1-randnumbersparent)+1))/2))
        children = parents[parentindices,:]
        # perform crossover
        crossoverpoint = np.random.rand(numpairs)*3
        crossoverindex = np.int64(np.floor(crossoverpoint))
        crossovervalue = crossoverpoint - crossoverindex
        for n in range(numpairs):
            originalchild1 = children[2*n,:]
            originalchild2 = children[2*n+1,:]
            ind=crossoverindex[n]
            val=crossovervalue[n]
            newchild1 = np.hstack((originalchild1[0:ind],originalchild2[ind:]))
            newchild2 = np.hstack((originalchild2[0:ind],originalchild1[ind:]))
            newchild1[ind]= originalchild1[ind]*val+originalchild2[ind]*(1-val)
            newchild2[ind]= originalchild2[ind]*val+originalchild1[ind]*(1-val)
            newchild1[ind]=np.maximum(np.minimum(newchild1[ind],1),0)
            newchild2[ind]=np.maximum(np.minimum(newchild2[ind],1),0)
            #np.clip(newchild1[ind], 0, 1, out=newchild1[ind])
            #np.clip(newchild2[ind], 0, 1, out=newchild2[ind])
            children[2*n,:]=newchild1
            children[2*n+1,:]=newchild2
        return children

    def applymutations(gatable,numelites):
        """
        Apply mutations to the genetic table.

        Parameters:
            - gatable (np.ndarray): Genetic table containing gene values.
            - numelites (int): Number of elite individuals.

        Returns:
            - np.ndarray: Genetic table with mutations applied.
        """
        shape_gatable = gatable.shape
        mutationhalfstepsize = 0.15
        mutationflag = np.less_equal(np.random.rand(shape_gatable[0],shape_gatable[1]),mutationrate)
        mutationvalues = np.random.uniform(-mutationhalfstepsize,mutationhalfstepsize,(shape_gatable[0],shape_gatable[1]))*mutationflag
        mutationvalues[0:numelites,:] = 0 #elite individuals are not mutated
        gatable = gatable + mutationvalues
        np.clip(gatable, 0, 1, out=gatable)    
        return gatable

    dirpath = './'
    datapath = './'
    outpath = './'
    #dataset_file = dirpath + 'all_struc_features.txt'
    model_file = input_files['xgb_model']
    data_filename = input_files['exp_data']
    q, input_data, err_data, err_q = np.loadtxt(data_filename, unpack = 'True', usecols = (0, 1, 2, 3))

    f_range_low = 0.5*input_data[0] # Lower range for scaling factor
    f_range_high = 2.0*input_data[0] # Higher range for scaling factor

    c_range_low = -2.0*input_data[-1] # Lower range for constant parameter.
    c_range_high = 2.0*input_data[-1] # Higher range for constant parameter.

    numq = len(q)
    xgboost_model = xgb.Booster(model_file=model_file)
    print('All models loaded')

    num_ga_runs = int(GA_res_params['num_ga_runs'])

    num_res_points = GA_res_params['num_res_points']
    numq_res       = int(num_res_points*numq)
    q_res          = np.zeros(numq_res)
    res_func       = np.zeros(numq_res)

    count = 0

    for qi in range(numq):

        del_q = err_q[qi]
        pts_x = np.linspace(q[qi] - 3*del_q, q[qi] + 3*del_q, num_res_points)
        pts_y = np.zeros(num_res_points)

        for i in range(num_res_points):
            pts_y[i] = (pts_x[i]/del_q**2)*np.exp(-0.5*(pts_x[i]**2 + q[qi]**2.0)/del_q**2)*(i0(q[qi]*pts_x[i]/del_q**2))
            #pts_y[i] = (1/(del_q * np.sqrt(2*np.pi)))*np.exp(-(pts_x[i] - q[qi])**2.0/(2*(del_q)**2.0))
            q_res[count] = pts_x[i]
            res_func[count] = pts_y[i]
            count += 1

    weights = np.zeros(numq)

    for i in range(1, numq):
        weights[i] = np.log(q[i]/q[i-1])

    weights[0] = weights[1]
    weights = weights/np.sum(weights)

    output_column_names = ['Rcore', 'LVF', 'Ndom', 'SLDL', 'Scaling_Fac', 'Const_Para', 'X2']
    output_ga_table_name = outfile_names['struc_feat_table']
    output_x2_iq_profile_name = outfile_names['best_x2_iq']

    #Generate Initial Population
    ipopsize=GA_res_params['ini_popsize']
    popsize=GA_res_params['popsize']
    numgenes=6

    ga_struc_features_table = np.zeros((int(popsize*num_ga_runs), int(numgenes+1)))
    iq_all_x2_best = np.zeros((numq, num_ga_runs))
    x2_best     = np.zeros(num_ga_runs) 

    iq_x2_best_performing = np.zeros(numq)
    std_err_mean_x2_iq    = np.zeros(numq)


    fmt = "%.10f %.10f %.10f\n"
    iq_out = []

    for run in range(num_ga_runs):

        print('Collecting CREASE-GA results for trial number: %d' %(run + 1))

        gatable = np.random.rand(ipopsize, numgenes + 1)
        currentprofiles = generateallprofiles(gatable)
        print('Initial profiles generated')
        gatable = updateallfitnesses(gatable, currentprofiles)
        print('Initial fitness evaluated')
        tableindices = gatable[:,numgenes].argsort() #sort by the descending fitness value
        gatable = gatable[tableindices[0:popsize]]
        print('Initial Populations defined')

        # GA steps
        numgens=GA_res_params['num_gen']
        numparents=30 # keep 30% of the population for mating
        numelites=2
        mutationrate=0.1

        meanfitness = np.mean(gatable[:, 6])
        stddevfitness = np.std(gatable[:,6])
        bestfitness = gatable[0,6]
        worstfitness = gatable[-1,6]
        diversitymetric = np.mean(np.sum((gatable-np.mean(gatable,axis=0))**2,axis=1))
        print('Generation: '+ str(0) +'. Lowest X2: ' + str(bestfitness) + '. Average X2: ' + str(np.mean(gatable[:, 6])) + '.')
        #print('The diversity metric is '+str(diversitymetric))

        #fitness scores initialization
        fitness_scores = np.array([[0,meanfitness,stddevfitness,bestfitness,worstfitness]])

        #Evolutionary process loop
        struc_featurestable = genes_to_struc_features(gatable[:,0:4])
        struc_featurestable = np.vstack((np.ones([1,4])*0,struc_featurestable))
        evolvedstrucfeatures=struc_featurestable.reshape((popsize+1,1,4))

        for currentgen in range(1,numgens+1):
            parents = gatable[0:numparents,:]
            children = generate_children(parents)
            gatable = np.vstack((parents,children))
            gatable = applymutations(gatable,numelites)
            currentprofiles = generateallprofiles(gatable)
            gatable = updateallfitnesses(gatable, currentprofiles)
            tableindices = gatable[:,numgenes].argsort() #sort by the descending fitness value
            gatable = gatable[tableindices]
            meanfitness = np.mean(gatable[:,6])
            stddevfitness = np.std(gatable[:,6])
            bestfitness = gatable[0,6]
            worstfitness = gatable[-1,6]
            diversitymetric = np.mean(np.sum((gatable-np.mean(gatable,axis=0))**2,axis=1))
            fitness_scores=np.append(fitness_scores,[[currentgen,meanfitness,stddevfitness,bestfitness,worstfitness]],axis=0)
            mutationrate = -np.log10(diversitymetric)*0.1
            if currentgen%10 == 0:
                print('Generation: '+ str(currentgen) +'. Lowest X2: ' + str(bestfitness) + '. Average X2: ' + str(np.mean(gatable[:, 6])) + '.')
                struc_featurestable = genes_to_struc_features(gatable[:,0:4]) 	
                struc_featurestable = np.vstack((np.ones([1,4])*currentgen,struc_featurestable))
                evolvedstrucfeatures=np.hstack((evolvedstrucfeatures,struc_featurestable.reshape((popsize+1,1,4))))

        struc_featurestable = genes_to_struc_features(gatable[:, 0:4])
        ga_struc_features_table[int(run*popsize):int((run+1)*popsize), 0:4] = struc_featurestable
        ga_struc_features_table[int(run*popsize):int((run+1)*popsize), 4] = gatable[:, 4]*(f_range_high - f_range_low) + f_range_low
        ga_struc_features_table[int(run*popsize):int((run+1)*popsize), 5] = gatable[:, 5]*(c_range_high - c_range_low) + c_range_low
        ga_struc_features_table[int(run*popsize):int((run+1)*popsize), 6] = gatable[:, 6]

        x2_best[run] = gatable[0, 6] 

        for qi in range(numq):

            iq_all_x2_best[qi, run] = (currentprofiles[qi, 0]/currentprofiles[0, 0])*(gatable[0, 4]*(f_range_high - f_range_low) + f_range_low) + gatable[0, 5]*(c_range_high - c_range_low) + c_range_low

    df_custom = pd.DataFrame(ga_struc_features_table, columns = output_column_names)
    df_custom.to_csv(output_ga_table_name)

    iq_out = []

    best_x2_ga_run = np.argmin(x2_best)
    iq_x2_best_performing = iq_all_x2_best[:, best_x2_ga_run]
    std_err_mean_x2_iq = np.std(iq_all_x2_best, axis = 1)

    for qi in range(numq):

        a = fmt % (q[qi], iq_x2_best_performing[qi], std_err_mean_x2_iq[qi])
        iq_out.append(a)

    open(output_x2_iq_profile_name, 'w').writelines(iq_out)

    print('X2 corresponding to best candidate solution', np.min(x2_best))
    print('Structural Features corresponding to Lowest X2:', ga_struc_features_table[int(best_x2_ga_run*popsize), 0:4])

    end_time = time.time()

    print('Total time for all GA runs in minutes:', (end_time - ini_time)/60)

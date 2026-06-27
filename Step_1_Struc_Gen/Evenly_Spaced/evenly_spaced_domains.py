########################################################################################################################

# Code created by Rohan S. Adhikari, Postdoctoral researcher, Jayaraman Lab, University of Delaware, Newark, DE.
# Code created in March 2026.
# Module for evenly spacing out isolated phase domains. Module called within inputscript_even_spaced.py
# Do not run this script, change parameters and file names in inputscript_even_spaced.py and run using ...
# ... python/python3 inputscript_even_spaced.py

########################################################################################################################

import numpy as np
import random
from scipy.special import erfcinv
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import time
import copy

def evenly_spaced_domains(params, file_names):

    ini_time = time.time()
    plt.style.use("paper.mplstyle")

    total_grids          = int(params['num_grids'])
    num_domains          = int(params['num_liq_dom'])

    num_phi_grid = int(np.sqrt(total_grids))
    num_thet_grid = int(np.sqrt(total_grids))

    num_bin_phi = int(np.sqrt(total_grids))
    num_bin_thet = int(np.sqrt(total_grids))

    fmt = "%d\n"


    outfile = file_names['out_grids_list']
    out_grids = []

    file_name = file_names['input_grid_file']
    dom_grids            = np.loadtxt(file_name)
    grids_per_entry  = len(dom_grids)


    num_grids_per_domain = np.zeros(num_domains, dtype = int)
    geometric_centers    = np.zeros(num_domains, dtype = int)
    lowest_thet          = np.zeros(num_domains, dtype = int)
    highest_thet         = np.zeros(num_domains, dtype = int)
    rem                  = int(grids_per_entry % num_domains)
    quot                 = int(grids_per_entry/num_domains)

    num_grids_per_domain[:] = quot 
    num_grids_per_domain[:rem] += 1

    if (rem == 0):
        ori_domains_thet = np.zeros((quot, num_domains))
        ori_domains_phi  = np.zeros((quot, num_domains))

        unbroken_domains_thet = np.zeros((quot, num_domains))
        unbroken_domains_phi = np.zeros((quot, num_domains))

        new_domains_thet = np.zeros((quot, num_domains))
        new_domains_phi  = np.zeros((quot, num_domains))

        temp_domains_thet = np.zeros((quot, num_domains))
        temp_domains_phi = np.zeros((quot, num_domains))

    else:

        ori_domains_thet = np.zeros((quot + 1, num_domains))
        ori_domains_phi  = np.zeros((quot + 1, num_domains))

        unbroken_domains_thet = np.zeros((quot + 1, num_domains))
        unbroken_domains_phi = np.zeros((quot + 1, num_domains))

        new_domains_thet = np.zeros((quot + 1, num_domains))
        new_domains_phi  = np.zeros((quot + 1, num_domains))

        temp_domains_thet = np.zeros((quot + 1, num_domains))
        temp_domains_phi = np.zeros((quot + 1, num_domains))

    count_grids = 0

    for dom in range(num_domains):
        for grid in range(num_grids_per_domain[dom]):

            ori_domains_thet[grid, dom] = int(dom_grids[int(count_grids)] / num_bin_phi)
            ori_domains_phi[grid, dom] = int(dom_grids[int(count_grids)] % num_bin_phi)
            count_grids += 1

    new_domains_thet = copy.deepcopy(ori_domains_thet)
    new_domains_phi = copy.deepcopy(ori_domains_phi)

    temp_domains_thet = copy.deepcopy(ori_domains_thet)
    temp_domains_phi = copy.deepcopy(ori_domains_phi)

    for dom in range(num_domains):
        for grid in range(num_grids_per_domain[dom]):

            if (grid == 0):

                unbroken_domains_thet[grid, dom] =  ori_domains_thet[grid, dom]
                unbroken_domains_phi[grid, dom] =  ori_domains_phi[grid, dom]

            else:

                unbroken_domains_thet[grid, dom] =  ori_domains_thet[grid, dom]

                if (int(ori_domains_thet[grid, dom] - ori_domains_thet[grid - 1, dom]) == num_bin_thet - 1):

                    unbroken_domains_phi[grid, dom] = unbroken_domains_phi[grid - 1, dom] - 1

                elif (int(ori_domains_thet[grid, dom] - ori_domains_thet[grid - 1, dom]) == -1*(num_bin_thet - 1)):

                    unbroken_domains_phi[grid, dom] = unbroken_domains_phi[grid - 1, dom] + 1

                else:

                    unbroken_domains_phi[grid, dom] = unbroken_domains_phi[grid-1, dom] + (ori_domains_thet[grid, dom] - ori_domains_thet[grid - 1, dom])

    for dom in range(num_domains):
        
        if (dom < rem):

            thet_min = np.min(unbroken_domains_thet[:, dom])
            thet_max = np.max(unbroken_domains_thet[:, dom]) 

            thet_cent = (thet_max - thet_min)/2

            phi_min = np.min(unbroken_domains_phi[:, dom])
            phi_max = np.max(unbroken_domains_phi[:, dom])

            phi_cent = (phi_max - phi_min)/2

        else:

            thet_min = np.min(unbroken_domains_thet[:-1, dom])
            thet_max = np.max(unbroken_domains_thet[:-1, dom]) 

            thet_cent = (thet_max - thet_min)/2

            phi_min = np.min(unbroken_domains_phi[:-1, dom])
            phi_max = np.max(unbroken_domains_phi[:-1, dom])

            phi_cent = (phi_max - phi_min)/2

        min_dist = 10**20

        for grid in range(num_grids_per_domain[dom]):

            min_dist_temp = (unbroken_domains_phi[grid, dom] - phi_cent)**2 + (unbroken_domains_thet[grid, dom] - thet_cent)**2     

            if (min_dist_temp < min_dist):

                min_dist = min_dist_temp
                geometric_centers[dom] = grid

        lowest_thet[dom] = int(unbroken_domains_thet[geometric_centers[dom], dom] - thet_min)
        highest_thet[dom] = int(num_bin_thet - (thet_max - unbroken_domains_thet[geometric_centers[dom], dom]))

    ini_dist_dom_cent = 0.0
    geom_thet_centers = np.zeros(num_domains)
    geom_phi_centers = np.zeros(num_domains)

    for dom in range(num_domains):

        geom_thet_centers[dom] = ori_domains_thet[geometric_centers[dom], dom]
        geom_phi_centers[dom] = ori_domains_phi[geometric_centers[dom], dom]

    geom_thet_centers = geom_thet_centers[np.argsort(geom_thet_centers)]
    geom_phi_centers = geom_phi_centers[np.argsort(geom_thet_centers)]

    dist_between_doms_thet = np.zeros(num_domains-1)
    dist_between_doms_phi = np.zeros(num_domains)

    for dom in range(0, num_domains-1):
        dist_between_doms_thet[dom] = geom_thet_centers[dom+1] - geom_thet_centers[dom]

    for dom in range(num_domains-1):
        dist_between_doms_phi[dom] = geom_phi_centers[dom+1] - geom_phi_centers[dom]
        #if (dist_between_doms_phi[dom] >= num_bin_phi/2):
        #    dist_between_doms_phi[dom] = num_bin_phi - 1 - dist_between_doms_phi[dom]

    dist_between_doms_phi[num_domains-1] = (geom_phi_centers[-1] - geom_phi_centers[0])

    #if (dist_between_doms_phi[num_domains-1] >= num_bin_phi):
    #    dist_between_doms_phi[num_domains-1] = num_bin_phi - 1 - dist_between_doms_phi[num_domains-1]

    max_diff = np.max(dist_between_doms_phi)

    if (np.max(dist_between_doms_thet) == 0):
        ini_dist_dom_thet = 0
    else:
        ini_dist_dom_thet = (geom_thet_centers[-1] - geom_thet_centers[0])/(num_bin_thet - 1) + (1 - np.std(dist_between_doms_thet)/np.max(dist_between_doms_thet)) 

    if (max_diff == 0):
        ini_dist_dom_phi = 0
    else:
        ini_dist_dom_phi = (max_diff)/(num_bin_phi) + (1 - np.std(dist_between_doms_phi)/max_diff)

    ini_dist_dom_cent = ini_dist_dom_thet + ini_dist_dom_phi

    print('Intital distance between grid centers', ini_dist_dom_cent)

    count = 0
    count_succ = 0

    for trial in range(10):

        for dom in range(num_domains):

            for thet in range(lowest_thet[dom], highest_thet[dom]):
                for phi in range(num_bin_phi):

                    grid_succ = True

                    temp_domains_thet = copy.deepcopy(new_domains_thet)
                    temp_domains_phi = copy.deepcopy(new_domains_phi)

                    diff_thet = new_domains_thet[geometric_centers[dom], dom] - thet
                    diff_phi  = new_domains_phi[geometric_centers[dom], dom] - phi

                    temp_domains_thet[:, dom] = temp_domains_thet[:, dom] - diff_thet
                    temp_domains_phi[:, dom] = temp_domains_phi[:, dom] - diff_phi    
                    temp_domains_phi[:, dom] = temp_domains_phi[:, dom] - np.floor(temp_domains_phi[:, dom]/num_bin_phi)*num_bin_phi

                    for grids in range(num_grids_per_domain[dom]):

                        if (grid_succ):

                            for prev_domain in range(dom):
                                for prev_grid in range(num_grids_per_domain[prev_domain]):

                                    cos_thet_bin_diff = int(np.absolute(temp_domains_thet[grids, dom] - temp_domains_thet[prev_grid, prev_domain]))
                                    phi_bin_diff = int(np.absolute(temp_domains_phi[grids, dom] - temp_domains_phi[prev_grid, prev_domain]))

                                    if (cos_thet_bin_diff <= 1 and (phi_bin_diff <= 1 or phi_bin_diff == int(num_phi_grid - 1))):

                                        grid_succ = False
                                        count += 1
                                        break

                                if (grid_succ):
                                    pass
                                else:
                                    break

                            if (grid_succ):

                                for next_domain in range(dom+1, num_domains):
                                    for next_grid in range(num_grids_per_domain[next_domain]):
                                            
                                        cos_thet_bin_diff = int(np.absolute(temp_domains_thet[grids, dom] - temp_domains_thet[next_grid, next_domain]))
                                        phi_bin_diff = int(np.absolute(temp_domains_phi[grids, dom] - temp_domains_phi[next_grid, next_domain]))

                                        if (cos_thet_bin_diff <= 1 and (phi_bin_diff <= 1 or phi_bin_diff == int(num_bin_phi - 1))):

                                            grid_succ = False
                                            count += 1
                                            break

                                    if (grid_succ):
                                        pass
                                    else:
                                        break

                    if(grid_succ):

                        loop_dist_dom_cent = 0.0

                        geom_thet_centers = np.zeros(num_domains)
                        geom_phi_centers = np.zeros(num_domains)

                        for iter_dom in range(num_domains):

                            geom_thet_centers[iter_dom] = temp_domains_thet[geometric_centers[iter_dom], iter_dom]
                            geom_phi_centers[iter_dom] = temp_domains_phi[geometric_centers[iter_dom], iter_dom]

                        geom_thet_centers = geom_thet_centers[np.argsort(geom_thet_centers)]
                        geom_phi_centers = geom_phi_centers[np.argsort(geom_thet_centers)]

                        dist_between_doms_thet = np.zeros(num_domains-1)
                        dist_between_doms_phi = np.zeros(num_domains)

                        for iter_dom in range(0, num_domains-1):
                            dist_between_doms_thet[iter_dom] = geom_thet_centers[iter_dom+1] - geom_thet_centers[iter_dom]

                        for iter_dom in range(0, num_domains-1):
                            dist_between_doms_phi[iter_dom] = geom_phi_centers[iter_dom+1] - geom_phi_centers[iter_dom]
                        #    if (dist_between_doms_phi[iter_dom] >= num_bin_phi/2):
                        #        dist_between_doms_phi[iter_dom] = num_bin_phi - 1 - dist_between_doms_phi[iter_dom]

                        dist_between_doms_phi[num_domains-1] = (geom_phi_centers[-1] - geom_phi_centers[0])

                        #if (dist_between_doms_phi[num_domains-1] >= num_bin_phi/2):
                        #    dist_between_doms_phi[num_domains-1] = num_bin_phi -1 - dist_between_doms_phi[num_domains-1]

                        max_diff = np.max(dist_between_doms_phi)

                        if (np.max(dist_between_doms_thet) == 0):
                            loop_dist_dom_thet = 0
                        else:
                            loop_dist_dom_thet = (geom_thet_centers[-1] - geom_thet_centers[0])/(num_bin_thet - 1) + (1 - np.std(dist_between_doms_thet)/np.max(dist_between_doms_thet)) 
                        
                        if (max_diff == 0):
                            loop_dist_dom_phi = 0.0
                        else:
                            loop_dist_dom_phi = (max_diff)/num_bin_phi + (1 - np.std(dist_between_doms_phi)/max_diff)

                        loop_dist_dom_cent = loop_dist_dom_thet + loop_dist_dom_phi
                        
                        if (loop_dist_dom_cent > ini_dist_dom_cent):
                                
                            count_succ += 1

                            new_domains_phi = copy.deepcopy(temp_domains_phi)
                            new_domains_thet = copy.deepcopy(temp_domains_thet)

                            ini_dist_dom_cent = loop_dist_dom_cent

    print('Final distance between grid centers', ini_dist_dom_cent)
    print('Number of grid moves success:', count_succ)

    for domain in range(num_domains):            
        for grid in range(num_grids_per_domain[domain]):
                
            a = fmt % (new_domains_thet[grid, domain]*num_phi_grid + new_domains_phi[grid, domain])
            out_grids.append(a)

    open(outfile, 'w').writelines(out_grids)

    out_grids_array = np.array(out_grids, dtype = int)
    num_liq_grids = len(out_grids_array)

#    print('Total number of liquid grids:', num_liq_grids)

#    print(num_bin_thet, num_bin_phi)

    out_grids_array = np.array(out_grids, dtype = int)
    num_liq_grids = len(out_grids_array)

#    print('Total number of liquid grids:', num_liq_grids)

#    print(num_bin_thet, num_bin_phi)

    data = {r'$0$':0, r'$\pi/2$':1.570796327, r'$\pi$':3.141592654, r'$3\pi/2$':4.71238898, r'$2\pi$':6.283185307}
    xtick_labels = list(data.keys())

    phi_start = np.pi/num_bin_phi
    phi_end = 2*np.pi - np.pi/num_bin_phi

    cos_thet_start = -1.0 + 1/num_bin_thet
    cos_thet_end = 1.0 - 1/num_bin_thet

    phi      = np.linspace(phi_start, phi_end, num_bin_phi)
    cos_thet = np.linspace(cos_thet_start, cos_thet_end, num_bin_thet)

    xticks = np.array([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])

    phi_grid, costhet_grid = np.meshgrid(phi, cos_thet)
    count_grids   = np.zeros((num_bin_thet, num_bin_thet))

    colors = ['xkcd:Light Blue Grey', 'blue']
    cmap = ListedColormap(colors)

    for grid in range(num_liq_grids):

        grid_phi = int(out_grids_array[grid] % num_bin_phi)
        grid_thet = int(out_grids_array[grid] / num_bin_thet)
        count_grids[grid_thet, grid_phi] += 1

    fig = plt.subplots(figsize=(7,7))
    plt.pcolormesh(phi_grid, costhet_grid, count_grids, cmap=cmap, edgecolors='k')
    plt.xlabel(r'\boldmath$\mathrm{\phi}$')
    plt.ylabel(r'\boldmath$\mathrm{\cos \theta}$')
    plt.yticks([-0.9, -0.3, 0.3, 0.9])
    plt.xticks(ticks = xticks, labels = xtick_labels)
    plt.savefig(file_names['out_flattened_rep'], dpi = 300)
    plt.tight_layout()


    point_scatterers_per_grid = int(params['num_point_scatterers']/total_grids)

#    print('Number of point scatterers per grid', point_scatterers_per_grid)

    fmt = "%d %d %.6f %.6f %.6f\n"
    out_1 = []
    out_2 = []

    bin_size_cos_thet = 2/num_bin_thet
    bin_size_phi      = 2*np.pi/num_bin_phi

    out_ps_dump = file_names['out_ps_dump_file']

    count_num_pts_1 = int(0)
    count_num_pts_2 = int(0)

    grids_patch_no  = []

    boxlen = params['boxlength']
    total_vesc_rad = params['vesc_core_rad'] + params['vesc_shell_thick']
    vesc_core_rad = params['vesc_core_rad']

    for bin_num in range(total_grids):

        patch_yes = False
				
        for patch_num in range(num_liq_grids):
            if (out_grids_array[patch_num] == bin_num):
                patch_yes = True
                break

        if (patch_yes):
            pass

        else:
            grids_patch_no.append(bin_num)

    grids_patch_no_int = np.array(grids_patch_no, dtype = int)

    len_grids_patch_yes = len(out_grids_array)
    len_grids_patch_no = len(grids_patch_no_int)

#    print(len_grids_patch_yes, len_grids_patch_no)

    for no_grids in range(len_grids_patch_no):
        for pts in range(point_scatterers_per_grid):

            grid_num = grids_patch_no_int[no_grids]
            u = random.random()
            cos_thet_low = -1 + bin_size_cos_thet*int(grid_num / num_bin_phi)
            cos_thet_high = -1 + bin_size_cos_thet*int(grid_num / num_bin_phi + 1)
            theta = np.arccos(cos_thet_low + random.random()*(cos_thet_high - cos_thet_low))
            phi_low = bin_size_phi*(grid_num%num_bin_phi)
            phi_high = bin_size_phi*(grid_num%num_bin_phi + 1)
            phi = phi_low + random.random()*(phi_high - phi_low)

            rad = (u*(total_vesc_rad**3.0 - vesc_core_rad**3.0) + vesc_core_rad**3.0)**(1/3)

            pts_x = rad*np.sin(theta)*np.cos(phi)
            pts_y = rad*np.sin(theta)*np.sin(phi)
            pts_z = rad*np.cos(theta)
    
            count_num_pts_1 += 1
            a = fmt % (count_num_pts_1, 1, pts_x, pts_y, pts_z)
            out_1.append(a)

    for yes_grid in range(len_grids_patch_yes):
        for pts in range(point_scatterers_per_grid):
            
            grid_num = out_grids_array[yes_grid]
            u = random.random()
            cos_thet_low = -1 + bin_size_cos_thet*int(grid_num / num_bin_phi)
            cos_thet_high = -1 + bin_size_cos_thet*int(grid_num / num_bin_phi + 1)
            theta = np.arccos(cos_thet_low + random.random()*(cos_thet_high - cos_thet_low))
            phi_low = bin_size_phi*(grid_num%num_bin_phi)
            phi_high = bin_size_phi*(grid_num%num_bin_phi + 1)
            phi = phi_low + random.random()*(phi_high - phi_low)

            rad = (u*(total_vesc_rad**3.0 - vesc_core_rad**3.0) + vesc_core_rad**3.0)**(1/3)

            pts_x = rad*np.sin(theta)*np.cos(phi)
            pts_y = rad*np.sin(theta)*np.sin(phi)
            pts_z = rad*np.cos(theta)

            count_num_pts_2 += 1
            a = fmt % (count_num_pts_2, 2, pts_x, pts_y, pts_z)
            out_2.append(a)		

    file_1 = open(out_ps_dump, 'w')
    file_1.write('ITEM: TIMESTEP\n')
	
    file_1 = open(out_ps_dump, 'a')
    file_1.write('0\n')

    file_1 = open(out_ps_dump, 'a')
    file_1.write('ITEM: NUMBER OF ATOMS\n')

    file_1 = open(out_ps_dump, 'a')
    file_1.write('%d\n' %(count_num_pts_1 + count_num_pts_2))

    file_1 = open(out_ps_dump, 'a')
    file_1.write('ITEM: BOX BOUNDS pp pp pp\n')

    file_1 = open(out_ps_dump, 'a')
    file_1.write('%.6f %.6f\n' %(-boxlen/2, boxlen/2))

    file_1 = open(out_ps_dump, 'a')
    file_1.write('%.6f %.6f\n' %(-boxlen/2, boxlen/2))

    file_1 = open(out_ps_dump, 'a')
    file_1.write('%.6f %.6f\n' %(-boxlen/2, boxlen/2))

    file_1 = open(out_ps_dump, 'a')
    file_1.write('ITEM: ATOMS id type x y z\n')

    file_1 = open(out_ps_dump, 'a')
    file_1.writelines(out_1)

    file_1 = open(out_ps_dump, 'a')
    file_1.writelines(out_2)

    file_1.close()

    print('Point scatterers file written')

    end_time = time.time()
    print('Total for structure generation in seconds:', end_time - ini_time)

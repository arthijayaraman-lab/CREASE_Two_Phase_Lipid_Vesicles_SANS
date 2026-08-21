################################################################################################################

# Module for performing constrained random walks in spherical grids. Created by Rohan S. Adhikari, 
# Postdoctoral Researcher, Jayaraman Lab, University of Delaware, Newark, DE, December 2025.
# Do not run this module by itself. Module is called within inputscript_crwsg.py.
# Set the required values for parameters and file names in inputscript_crwsg.py and run using ...
# ... python/python3 inputscript_crwsg.py.  

################################################################################################################

import numpy as np
import random 
from scipy.special import erfcinv
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import time

def VISIT(params, file_names):

    plt.style.use("paper.mplstyle")

    ini_time = time.time()

    max_overlap  = 1000000
    max_transl   = 10000

    max_int_no_inv = int(724)
    max_int_rw     = int(150)

    max_trials = 1001
    max_trials_2 = 100000

    tot_bins  = int(params['num_grids'])

    max_int_no_inv = int(0.7*tot_bins)
    max_int_rw     = int(0.15*tot_bins)

    num_bin_cos_thet = int(np.sqrt(tot_bins))
    num_bin_phi = int(np.sqrt(tot_bins))

    fmt_g = "%d"
    fmt_w = "%d\n"
    out_grids_all = []
    out_vesc = []
    random_walk_fail = True
    tot_bins_patch = int(params['lvf']*tot_bins)

    print('Performing random walk for %d liquid bins on %d total bins' %(tot_bins_patch, tot_bins))

    inv_patch = False

    if (tot_bins_patch > max_int_no_inv):
        tot_bins_patch = int(tot_bins - tot_bins_patch)
        inv_patch = True

    all_bins_patch = tot_bins_patch

    if (tot_bins_patch > max_int_rw):
        tot_bins_patch = max_int_rw

    num_domains = int(params['num_liq_dom'])

    while (random_walk_fail):

        grids_patch_yes = []
        normal_random_walk_fail = True
        num_bins_patch      = np.zeros(num_domains, dtype = int)
        patches_domain_quot = int(tot_bins_patch/num_domains)
        patches_domain_rem  = int(tot_bins_patch % num_domains) 

        for domain in range(num_domains):

            if (domain < patches_domain_rem):

                num_bins_patch[domain] = patches_domain_quot + 1

            else:

                num_bins_patch[domain] = patches_domain_quot
                
        max_domains = int(np.max(num_bins_patch))

        bin_start = []

        ini_all_patches = []

        for bin_num in range(tot_bins):

            a = fmt_g % (bin_num)
            ini_all_patches.append(a)

        num_non_overlap = len(ini_all_patches)
        bin_grids_patch = np.zeros((max_domains, num_domains, 2), dtype = int)

        for domain in range(num_domains):

            ini_domain_fail = True

            while(ini_domain_fail):
                
                bin_next = False

                start_bin_num = int(random.random()*(num_non_overlap-1))
                bin_grids_patch[0, domain, 0] = int(int(ini_all_patches[start_bin_num])/num_bin_cos_thet)
                bin_grids_patch[0, domain, 1] = int(int(ini_all_patches[start_bin_num]) % num_bin_cos_thet)

                del ini_all_patches[start_bin_num]
                num_non_overlap = len(ini_all_patches)

                for prev_domain in range(domain):

                    cos_thet_bin_diff = int(np.absolute(bin_grids_patch[0, domain, 0] - bin_grids_patch[0, prev_domain, 0]))
                    phi_bin_diff = int(np.absolute(bin_grids_patch[0, domain, 1] - bin_grids_patch[0, prev_domain, 1]))
                
                    if (cos_thet_bin_diff <= 1 and (phi_bin_diff <= 1 or phi_bin_diff == int(num_bin_phi - 1))):
                    
                        bin_next = True
                        break

                if (bin_next):
                    pass

                else:
                    ini_domain_fail = False    

        break_loop = False

        for domain in range(num_domains):

            if (domain == num_domains - 1 and num_bins_patch[domain] <= 1) :

                normal_random_walk_fail = False

            for grids in range(1, num_bins_patch[domain]):
                
                for trials in range(max_trials):

                    grid_succ = True
                    same     = False

                    if (random.random() < 1/3):
                    
                        bin_grids_patch[grids, domain, 0] = int(bin_grids_patch[grids -1, domain, 0] - 1)

                        if (bin_grids_patch[grids, domain, 0] == -1):

                            bin_grids_patch[grids, domain, 0] = bin_grids_patch[grids - 1, domain, 0]
                            same = True


                    elif (random.random() < 1/2):

                        bin_grids_patch[grids, domain, 0] = int(bin_grids_patch[grids -1, domain, 0])
                        same = True

                    else:

                        bin_grids_patch[grids, domain, 0] = int(bin_grids_patch[grids -1, domain, 0] + 1)

                        if (bin_grids_patch[grids, domain, 0] == num_bin_cos_thet):

                            bin_grids_patch[grids, domain, 0] = bin_grids_patch[grids -1, domain, 0]
                            same = True

                
                    if (random.random() < 1/3):
                    
                        bin_grids_patch[grids, domain, 1] = int(bin_grids_patch[grids -1, domain, 1] - 1)

                        if (bin_grids_patch[grids, domain, 1] == -1):

                            bin_grids_patch[grids, domain, 1] = int(num_bin_phi - 1)
                            same = False

                    elif (random.random() < 1/2):

                        bin_grids_patch[grids, domain, 1] = int(bin_grids_patch[grids -1, domain, 1])

                    else:

                        bin_grids_patch[grids, domain, 1] = int(bin_grids_patch[grids -1, domain, 1] + 1)

                        if (bin_grids_patch[grids, domain, 1] == num_bin_phi):

                            bin_grids_patch[grids, domain, 1] = int(0)
                            same = False

                    if (same):
                    
                        grid_succ = False

                    else:

                        for prev_grid in range(grids):

                            if (bin_grids_patch[grids, domain, 0] == bin_grids_patch[prev_grid, domain, 0] and bin_grids_patch[grids, domain, 1] == bin_grids_patch[prev_grid, domain, 1]):

                                grid_succ = False
                                break

                        for prev_domain in range(domain):
                            for prev_grid in range(num_bins_patch[prev_domain]):

                                cos_thet_bin_diff = int(np.absolute(bin_grids_patch[grids, domain, 0] - bin_grids_patch[prev_grid, prev_domain, 0]))
                                phi_bin_diff = int(np.absolute(bin_grids_patch[grids, domain, 1] - bin_grids_patch[prev_grid, prev_domain, 1]))

                                if (cos_thet_bin_diff <= 1 and (phi_bin_diff <= 1 or phi_bin_diff == int(num_bin_phi - 1))):

                                    grid_succ = False
                                    break

                        for next_domain in range(domain+1, num_domains):
                            if (num_bins_patch[next_domain] > 0):
                                
                                cos_thet_bin_diff = int(np.absolute(bin_grids_patch[grids, domain, 0] - bin_grids_patch[0, next_domain, 0]))
                                phi_bin_diff = int(np.absolute(bin_grids_patch[grids, domain, 1] - bin_grids_patch[0, next_domain, 1]))

                                if (cos_thet_bin_diff <= 1 and (phi_bin_diff <= 1 or phi_bin_diff == int(num_bin_phi - 1))):

                                    grid_succ = False
                                    break

                    if (grid_succ):
                        break


                if (trials == max_trials - 1):

                    break_loop = True
                    break

                if (grids == num_bins_patch[domain] - 1  and domain == num_domains - 1):

                    normal_random_walk_fail = False


            if (break_loop):

                break

        if (normal_random_walk_fail):
            pass

        elif(all_bins_patch > max_int_rw):

            out_vesc = []

            for domain in range(num_domains):
                for grid in range(num_bins_patch[domain]):

                    a = fmt_g % (bin_grids_patch[grid, domain, 0]*num_bin_cos_thet + bin_grids_patch[grid, domain, 1])
                    out_vesc.append(a)
            
            num_bins_patch_all      = np.zeros(num_domains, dtype = int)
            patches_domain_quot_all = int(all_bins_patch/num_domains)
            patches_domain_rem_all  = int(all_bins_patch % num_domains) 

            for domain in range(num_domains):

                if (domain < patches_domain_rem_all):

                    num_bins_patch_all[domain] = patches_domain_quot_all + 1

                else:

                    num_bins_patch_all[domain] = patches_domain_quot_all

            max_domains_all     = int(np.max(num_bins_patch_all))
            bin_grids_patch_all = np.zeros((max_domains_all, num_domains, 2), dtype = int)

            bin_grids_patch_no = []

            for patch in range(1024):

                patch_yes = False
            
                for patch_done in range(tot_bins_patch):

                    if (int(out_vesc[patch_done]) == patch):

                        patch_yes = True
                        break

                if (patch_yes):
                    pass
                else:
                    bin_grids_patch_no.append(patch)
            
            
            bin_grids_patch_no_int = [int(item) for item in bin_grids_patch_no]
            num_grids_patch_no = len(bin_grids_patch_no_int)

            for domain in range(num_domains):
                
                patch_next = True
                
                bin_grids_patch_all[:num_bins_patch[domain], domain, 0] = bin_grids_patch[:num_bins_patch[domain], domain, 0]
                bin_grids_patch_all[:num_bins_patch[domain], domain, 1] = bin_grids_patch[:num_bins_patch[domain], domain, 1]

                if (domain == (num_domains - 1) and num_bins_patch[domain] == num_bins_patch_all[domain]):

                    random_walk_fail = False

                for patch in range(num_bins_patch[domain], num_bins_patch_all[domain]):

                    num_grids_patch_no             = len(bin_grids_patch_no_int)
                    fr_next_bin_grids_patch_no_int = [] 
                    fr_next_num_grids_patch_no     = num_grids_patch_no

                    for create_nxt in range(num_grids_patch_no):

                        fr_next_bin_grids_patch_no_int.append(bin_grids_patch_no_int[create_nxt])

                    for fr_next in range(fr_next_num_grids_patch_no):

                        num_patch_no  = int(fr_next_num_grids_patch_no - fr_next)

                        patch_num      = fr_next_bin_grids_patch_no_int[int(random.random()*(num_patch_no-1))]
                        patch_num_phi  = int(patch_num % num_bin_cos_thet)
                        patch_num_thet = int(patch_num / num_bin_cos_thet)  
                        patch_next     = False
                        patch_iso      = True
                        fr_next_bin_grids_patch_no_int.remove(patch_num)                            

                        for prev_domain in range(domain):
                            for grids in range(num_bins_patch_all[prev_domain]):

                                patch_yes_phi  = bin_grids_patch_all[grids, prev_domain, 1]
                                patch_yes_thet = bin_grids_patch_all[grids, prev_domain, 0]

                                abs_patch_phi_diff  = np.absolute(patch_yes_phi - patch_num_phi)
                                abs_patch_thet_diff = np.absolute(patch_yes_thet - patch_num_thet) 

                                if ((abs_patch_phi_diff <= 1 or abs_patch_phi_diff == (num_bin_phi - 1)) and (abs_patch_thet_diff <= 1)):
                                    patch_iso = False
                                    break

                            if (patch_iso):
                                pass
                            else:
                                break

                        for next_domain in range(domain+1, num_domains):
                            for grids in range(num_bins_patch[next_domain]):

                                patch_yes_phi  = bin_grids_patch[grids, next_domain, 1]
                                patch_yes_thet = bin_grids_patch[grids, next_domain, 0]

                                abs_patch_phi_diff  = np.absolute(patch_yes_phi - patch_num_phi)
                                abs_patch_thet_diff = np.absolute(patch_yes_thet - patch_num_thet) 

                                if ((abs_patch_phi_diff <= 1 or abs_patch_phi_diff == (num_bin_phi - 1)) and (abs_patch_thet_diff <= 1)):
                                    patch_iso = False
                                    break

                            if (patch_iso):
                                pass
                            else:
                                break

                    
                        if (patch_iso):
                            for yes_patch in range(patch):

                                patch_yes_phi  = bin_grids_patch_all[yes_patch, domain, 1]
                                patch_yes_thet = bin_grids_patch_all[yes_patch, domain, 0]

                                abs_patch_phi_diff  = np.absolute(patch_yes_phi - patch_num_phi)
                                abs_patch_thet_diff = np.absolute(patch_yes_thet - patch_num_thet)

                                if ((abs_patch_phi_diff <= 1 or abs_patch_phi_diff == (num_bin_phi - 1)) and (abs_patch_thet_diff <= 1)):

                                    patch_next = True
                                    a = fmt_g % (patch_num)
                                    bin_grids_patch_all[patch, domain, 0] = patch_num_thet
                                    bin_grids_patch_all[patch, domain, 1] = patch_num_phi
                                    bin_grids_patch_no_int.remove(patch_num)
                                    break


                            if (patch_next):
                                break

                    if (patch_next):
                        if (domain == (num_domains - 1) and patch == (num_bins_patch_all[domain] - 1)):
                            random_walk_fail = False

                    if patch_next == False and (fr_next == (num_grids_patch_no - 1)):
                        break

                if patch_next == False and (fr_next == (num_grids_patch_no - 1)):
                        break

        else:
            random_walk_fail = False



    if (all_bins_patch <= max_int_rw):

        bin_grids_patch_all = bin_grids_patch
        num_bins_patch_all = num_bins_patch

    for domain in range(num_domains):
        
        for grid in range(num_bins_patch_all[domain]):
        
            a = fmt_w % (bin_grids_patch_all[grid, domain, 0]*num_bin_cos_thet + bin_grids_patch_all[grid, domain, 1])
            out_grids_all.append(a)

    open(file_names['out_grids_list'], 'w').writelines(out_grids_all)

    print('Grid List written')

    out_grids_array = np.array(out_grids_all, dtype = int)
    num_liq_grids = len(out_grids_all)

#    print('Total number of liquid grids:', num_liq_grids)

#    print(num_bin_cos_thet, num_bin_phi)

# Create flattened representation image.

    data = {r'$0$':0, r'$\pi/2$':1.570796327, r'$\pi$':3.141592654, r'$3\pi/2$':4.71238898, r'$2\pi$':6.283185307}
    xtick_labels = list(data.keys())

    phi_start = np.pi/num_bin_phi
    phi_end = 2*np.pi - np.pi/num_bin_phi

    cos_thet_start = -1.0 + 1/num_bin_cos_thet
    cos_thet_end = 1.0 - 1/num_bin_cos_thet

    phi      = np.linspace(phi_start, phi_end, num_bin_phi)
    cos_thet = np.linspace(cos_thet_start, cos_thet_end, num_bin_cos_thet)

    xticks = np.array([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])

    phi_grid, costhet_grid = np.meshgrid(phi, cos_thet) 
    count_grids   = np.zeros((num_bin_cos_thet, num_bin_cos_thet))

    colors = ['xkcd:Light Blue Grey', 'blue']
    cmap = ListedColormap(colors)

    for grid in range(num_liq_grids):

        grid_phi = int(out_grids_array[grid] % num_bin_phi)
        grid_thet = int(out_grids_array[grid] / num_bin_cos_thet)
        count_grids[grid_thet, grid_phi] += 1

    fig = plt.subplots(figsize=(7,7))
    plt.pcolormesh(phi_grid, costhet_grid, count_grids, cmap=cmap, edgecolors='k')
    plt.xlabel(r'\boldmath$\mathrm{\phi}$')
    plt.ylabel(r'\boldmath$\mathrm{\cos \theta}$')
    plt.yticks([-0.9, -0.3, 0.3, 0.9])
    plt.xticks(ticks = xticks, labels = xtick_labels)
    plt.savefig(file_names['out_flattened_rep'], dpi = 300)
    plt.tight_layout()

    print('Flattened representation image created')

    point_scatterers_per_grid = int(params['num_point_scatterers']/tot_bins)

#    Creating point scatterers representation of the spherical vesicle.

    fmt = "%d %d %.6f %.6f %.6f\n"
    out_1 = []
    out_2 = []

    bin_size_cos_thet = 2/num_bin_cos_thet
    bin_size_phi      = 2*np.pi/num_bin_phi

    out_ps_dump = file_names['out_ps_dump_file']

    count_num_pts_1 = int(0)
    count_num_pts_2 = int(0)

    grids_patch_no  = []

    boxlen = params['boxlength']
    total_vesc_rad = params['vesc_core_rad'] + params['vesc_shell_thick']
    vesc_core_rad = params['vesc_core_rad']

    for bin_num in range(tot_bins):

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

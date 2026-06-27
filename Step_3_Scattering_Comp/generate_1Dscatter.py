# -*- coding: utf-8 -*-
"""
Modified from 1D_scatter.py to process large LAMMPS dump files in chunks
"""
import os
import numpy as np
import pandas as pd
import cupy as cp
from time import perf_counter


def fibonacci_sphere(samples=1000, direction=[0, 1, 0]):
    """
    Generate points on a sphere's surface using a Fibonacci spiral algorithm.
    This provides a uniform distribution of directions.
    """
    phi = cp.pi * (3. - cp.sqrt(5.))  # golden angle in radians
    if samples == 1:
        point = cp.array([direction])
        return point / cp.linalg.norm(point, axis=1)

    indices = cp.arange(samples)
    y = 1 - (indices / (samples - 1)) * 2
    radius = cp.sqrt(1 - y * y)
    theta = phi * indices
    x = cp.cos(theta) * radius
    z = cp.sin(theta) * radius
    points = cp.column_stack((x, y, z))
    return points

def single_loop_minus_box(qrange, len_box, box_shift, coords, apply_center_correction=False, direction=[0, 1, 0], total_points=300):
    """
    Compute 1D scattering profile from 3D structure.
    """
    
    points = cp.asarray(coords)  # shape (N, 3)
    v_array = cp.asarray(fibonacci_sphere(total_points, direction).T)  # shape (3, total_points)
    q_range_cp = cp.asarray(qrange)
    if apply_center_correction:
        points = points - cp.asarray(box_shift)

    num_points = points.shape[0]
    num_q = q_range_cp.size
    num_dirs = v_array.shape[1]

    ret = cp.zeros((num_q, num_dirs), dtype=cp.float32)

    start_time = perf_counter()
    batch_size = num_points
    num_batches = (num_points + batch_size - 1) // batch_size
    print(f"Number of batches: {num_batches}")
    sum_exp = cp.zeros((num_dirs,), dtype=cp.complex64)

    # Main loop over q
    for qi in range(num_q):
        q = q_range_cp[qi]
        q_vecs = q * v_array  # shape (3, num_dirs)

        sum_exp[:] = 0 + 0j

        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_points)
            r_batch = points[start:end]  # shape (batch_size, 3)
            rvqs = cp.matmul(r_batch, q_vecs)  # shape (batch_size, num_dirs)
            sum_exp += cp.exp(-1j * rvqs).sum(axis=0)

        # Box form factor
        qx = q_vecs[0]
        qy = q_vecs[1]
        qz = q_vecs[2]
        # Using np.sinc:
        box_ff = cp.sinc(qx * len_box[0] / (2 * cp.pi)) * \
                 cp.sinc(qy * len_box[1] / (2 * cp.pi)) * \
                 cp.sinc(qz * len_box[2] / (2 * cp.pi))

        sum_exp = sum_exp / num_points - box_ff
        intensity = cp.real(sum_exp * cp.conj(sum_exp))
        ret[qi, :] = intensity

    omega = ret.mean(axis=1)
    end_time = perf_counter()
    print("Elapsed time:", end_time - start_time, "seconds")
    return omega

def parse_lammps_header(file):
    """Parse only the header of a LAMMPS dump file to get number of atoms and box bounds"""
    number_of_atoms = None
    box_bounds = None
    
    while True:
        line = file.readline()
        if not line:
            break
        line_strp = line.strip()
        
        if line_strp.startswith("ITEM: NUMBER OF ATOMS"):
            num_atoms_line = file.readline().strip()
            number_of_atoms = int(num_atoms_line)

        elif line_strp.startswith("ITEM: BOX BOUNDS"):
            bounds = []
            for _ in range(3):
                bounds_line = file.readline().strip().split()
                bounds.append(list(map(float, bounds_line)))
            box_bounds = np.array(bounds)
            
        elif line_strp.startswith("ITEM: ATOMS"):
            break
            
    return number_of_atoms, box_bounds, file.tell()  # Return file position for atoms start

def process_large_file_in_batches(input_file, output_file, batch_size=500000, apply_center_correction=False):
    """
    Process a very large LAMMPS file by loading and computing in batches,
    then incrementally building the final result. Never stores entire dataset -- will accumulate results incrementally across batches.
    This is a more memory-efficient approach for large files.
    """
    with open(input_file, 'r') as file:
        # Parse header first to get atom count and box info
        number_of_atoms, box_bounds, atoms_start_pos = parse_lammps_header(file)
        
        if number_of_atoms is None or box_bounds is None:
            raise ValueError("Could not parse LAMMPS file header properly")
            
        print(f"Total atoms in file: {number_of_atoms}")
        print(f"Box bounds:\n{box_bounds}")
        
        # Calculate length of box
        len_box = box_bounds[:, 1] - box_bounds[:, 0]
        box_shift = len_box / 2.0
        
        # Prepare q-range
        q_range = cp.loadtxt('ML_Training_q_values.txt')
        
        # Initialize for accumulating results
        sum_exp_total = None
        atoms_processed = 0
        columns = ['id', 'mol', 'type', 'x', 'y', 'z']
        
        # Reset file position to start of atoms data
        file.seek(atoms_start_pos)
        
        # Setup for summing over all batches
        v_array = cp.asarray(fibonacci_sphere(300, [0, 1, 0]).T)  # shape (3, total_points)
        num_dirs = v_array.shape[1]
        num_q = len(q_range)
        ret = cp.zeros((num_q, num_dirs), dtype=cp.float32)
        
        # Process in batches
        batch_num = 0
        
        while atoms_processed < number_of_atoms:
            start_time = perf_counter()
            batch_atoms = []
            batch_size_actual = min(batch_size, number_of_atoms - atoms_processed)
            
            print(f"Processing batch {batch_num+1} with {batch_size_actual} atoms...")
            
            # Read batch_size atoms or remaining atoms
            for _ in range(batch_size_actual):
                try:
                    atom_line = file.readline().strip().split()
                    if not atom_line:  # In case we hit end of file unexpectedly
                        break
                    # Convert to float
                    atom_data = list(map(float, atom_line))
                    batch_atoms.append(atom_data)
                except Exception as e:
                    print(f"Error reading atom at line {atoms_processed + len(batch_atoms)}: {e}")
                    break
            
            if not batch_atoms:
                break
                
            # Convert batch to DataFrame
            batch_df = pd.DataFrame(batch_atoms, columns=columns)
            
            # Extract coordinates
            pts = np.column_stack((batch_df['x'].values, batch_df['y'].values, batch_df['z'].values))
            points = cp.asarray(pts)
            
            if sum_exp_total is None:
                # First batch, initialize the accumulator
                sum_exp_total = cp.zeros((num_q, num_dirs), dtype=cp.complex64)
            
            # Apply center correction if needed
            if apply_center_correction:
                points = points - cp.asarray(box_shift)
            
            # Main loop over q values
            for qi in range(num_q):
                q = q_range[qi]
                q_vecs = q * v_array  # shape (3, num_dirs)
                
                # Calculate contribution from this batch
                rvqs = cp.matmul(points, q_vecs)  # shape (batch_size, num_dirs)
                sum_exp_batch = cp.exp(-1j * rvqs).sum(axis=0)
                
                # Add to total
                sum_exp_total[qi] += sum_exp_batch
            
            atoms_processed += len(batch_atoms)
            batch_num += 1
            
            # Free memory
            del batch_df
            del batch_atoms
            del points
            cp.get_default_memory_pool().free_all_blocks()
            
            end_time = perf_counter()
            print(f"Batch {batch_num} processed in {end_time - start_time:.2f} seconds")
            print(f"Total atoms processed so far: {atoms_processed}/{number_of_atoms}")
       
        out = []
        fmt = "%.10f %.10f\n" 
        # Calculate final intensity
        for qi in range(num_q):
            q = q_range[qi]
            q_vecs = q * v_array
            
            # Box form factor
            qx = q_vecs[0]
            qy = q_vecs[1]
            qz = q_vecs[2]
            box_ff = cp.sinc(qx * len_box[0] / (2 * cp.pi)) * \
                     cp.sinc(qy * len_box[1] / (2 * cp.pi)) * \
                     cp.sinc(qz * len_box[2] / (2 * cp.pi))
            
            # Normalize and subtract box
            #sum_exp = sum_exp_total[qi] / number_of_atoms - box_ff
            sum_exp = sum_exp_total[qi] / number_of_atoms             
            sum_exp_real = sum_exp.real.reshape(num_dirs)
            sum_exp_imag = sum_exp.imag.reshape(num_dirs)
            #sum_exp_flatten = sum_exp.flatten()
            #print(sum_exp)
            
            for orient in range(num_dirs):
                a = fmt % (sum_exp_real[orient], sum_exp_imag[orient])
                out.append(a)
            
            intensity = cp.real(sum_exp * cp.conj(sum_exp))
            ret[qi, :] = intensity
        
        #print(sum_exp_real)
        # Average over all directions
        omega = ret.mean(axis=1)
        
        open(output_file, 'w').writelines(out) 
        # Write results to file
        #with open(output_file, 'w') as out_file:
        #    for i in range(len(q_range)):
        #        out_file.write(f"{float(q_range[i])} {float(omega[i])}\n")
        
        print(f"Processing complete. Results saved to {output_file}")
        
        # Free GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        return q_range, omega, number_of_atoms

if __name__ == '__main__':
    # Directory containing all of your .dump files
    input_dir = './output/'  # DON'T FORGET TO CHANGE THIS
    
    # Directory where you want to save all scatter outputs
    output_dir = './scatter_output/'  # DON'T FORGET TO CHANGE THIS
    os.makedirs(output_dir, exist_ok=True)
    
    # Set batch size based on your GPU memory
    batch_size = 100000 
    
    for filename in os.listdir(input_dir):
        if "scatterers" in filename:
            file_path = os.path.join(input_dir, filename)
            print(f"\nProcessing file: {file_path}")
            
            # Create output file path
            base_file_name = os.path.basename(filename)
            output_file_name = base_file_name.replace('_scatterers', '_1D_profile')
            output_file_path = os.path.join(output_dir, output_file_name)
            
            try:
                # Use the batched processing function
                process_large_file_in_batches(file_path, output_file_path, batch_size=batch_size)
                print(f"Finished processing {filename}. Output saved to {output_file_path}.")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                # If error occurs, try with a smaller batch size
                print("Trying again with smaller batch size...")
                try:
                    process_large_file_in_batches(file_path, output_file_path, batch_size=batch_size//2)
                    print(f"Succeeded with smaller batch size. Output saved to {output_file_path}.")
                except Exception as e2:
                    print(f"Failed again: {e2}")
                    continue

            # Clear GPU memory after each file
            cp.get_default_memory_pool().free_all_blocks()
            print(f"GPU memory cleared after processing {filename}")

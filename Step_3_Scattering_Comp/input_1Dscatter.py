# -*- coding: utf-8 -*-
"""
Created on 2025-05-10 13:39:16
@author: skronen and svrakepati 
"""

import os
import numpy as np
import pandas as pd
import cupy as cp
from generate_1Dscatter import *

# Directory containing all of your .dump files
input_dir = './output/'  

# Directory where you want to save all scatter outputs
output_dir = './scatter_output/'
os.makedirs(output_dir, exist_ok=True)

# Set batch size based on your GPU memory
batch_size = 100000 

for filename in os.listdir(input_dir):
    if "scatterers" in filename:
        file_path = os.path.join(input_dir, filename)
        print(f"\nProcessing file: {file_path}")
        
        # Create output file path
        base_file_name = os.path.basename(filename)
        output_file_name = base_file_name.replace('_scatterers', '_Aq_profile')
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

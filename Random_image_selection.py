# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:14:30 2021

@author: tajr
"""

import numpy as np
import os
import random
from shutil import copyfile

# set directories
directory = 'C:/Users/tajr/Desktop/Outside_Data/ClassSplit/U'
target_directory = 'Images'
data_set_percent_size = float(0.07)

#print(os.listdir(directory))

# list all files in dir that are an image
files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

#print(files)

# select a percent of the files randomly 
#random_files = random.sample(files, int(len(files)*data_set_percent_size))
#random_files = np.random.choice(files, int(len(files)*data_set_percent_size))
random_files = random.sample(files, 100)

#print(random_files)

# move the randomly selected images by renaming directory 

for random_file_name in random_files:      
    #print(directory+'/'+random_file_name)
    #print(target_directory+'/'+random_file_name)
    #os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    copyfile(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    continue

# move the relevant labels for the randomly selected images

#for image_labels in random_files:
#    # strip extension and add .txt to find corellating label file then rename directory. 
#    os.rename(directory+'/'+(os.path.splitext(image_labels)[0]+'.txt'), target_directory+'/'+(os.path.splitext(image_labels)[0]+'.txt'))
#
#    continue
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 19:19:03 2021

@author: tajr
"""

import os

folder = 'C:/Users/tajr/Desktop/Resize + crop + annotations'

files = []
file = ''

# r = root, d = directories, f = files
for r, d, f in os.walk(folder):
    for file in f:
        if '.jpg' in file:
            files.append(file)

for target_image in files:
    
    fileName, ext = os.path.splitext(target_image)
    txt = open(folder + '/' + fileName + '.txt',"w+")
    txt.close()


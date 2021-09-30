# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:35:51 2021

@author: tajr
"""

import os
import cv2
import numpy as np

inputFolder = 'Input_videos/'
OutputFolder = 'Ouput_videos/'
VidName = 'Test.mp4'

fps = 20
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

cap = cv2.VideoCapture(inputFolder + VidName)
out = cv2.VideoWriter(OutputFolder + VidName, fourcc, fps, (416,416))

while True:
    
    ret, frame = cap.read()
    if np.shape(frame) == (): break
    sky = frame[0:0 + 720, 300:300 + 720]
    sky = cv2.resize(sky, (416, 416))
    out.write(sky)
    #cv2.imshow('Video', sky)
    #k = cv2.waitKey(1)
    #if k == 27: break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done")




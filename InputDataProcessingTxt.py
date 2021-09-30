# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:13:26 2021

@author: tajr
"""

#---------------------------------------------------#
# File locations (sort out file names properly)
#---------------------------------------------------#

# Input images
input_Image = 'Images/'

# Ouput images
output_Image = 'Output_images/'

#
input_txt = 'Annotations/'

#
output_txt = 'Output_annotations/'

# Input XML
input_xml = 'Annotations/'

# Output XML
output_xml = 'Output_annotations/'

#---------------------------------------------------#
# Preprocessors
#---------------------------------------------------#

import os
import shutil
import cv2
import numpy as np

# Convert from Yolo_mark to opencv format
def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H

    voc = []

    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))

    return [int(v) for v in voc]

# convert from opencv format to yolo format
# H,W is the image height and width
def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]

    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2

    return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H,
                                                                                                            6)
        
class yoloRotatebbox:
    def __init__(self, filename, image_ext, angle):
        
        # Assert checks
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image().shape[:2]

        f = open(self.filename + '.txt', 'r')

        f1 = f.readlines()

        new_bbox = []

        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)

                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)

                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                 new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat

#---------------------------------------------------#
# User Declarations
#---------------------------------------------------#

# Reshape size
Rwidth = 416
Rheight = 416

# Crop size
Cwidth = 720
Cheight = 720

# Rotation angles (degrees)
Rot1 = 5
Rot2 = 10

# Mode Select: (Only use one at a time)

# Returns augmented images deifned below
data_augmentation = True

# Video Crop (crops imges to specific size)
image_crop = False

# Bounding box changes (if not normilised data)
bounding = False

# reshaping images (reshapes to width x height)
image_reshape = False

# If only rotation is wanted (because of bugs I don't hvae time to fix)
Rotations = False

#------------------------------------------------------#
# Finding images (save array of files in f)
#------------------------------------------------------#

files = []
file = ''

# r = root, d = directories, f = files
for r, d, f in os.walk(input_Image):
    for file in f:
        if '.jpg' in file:
            files.append(file)

# If not images are found
if not files:
    print('no images found')

# Making save dirrectory's
if not os.path.isdir(output_Image):
    os.makedirs(output_Image)

if not os.path.isdir(output_xml):
    os.makedirs(output_xml)

#------------------------------------------------------#
# Cropping images (set area to be cropped above)
#------------------------------------------------------#

num = '3_'

if image_crop:
    
    print('Cropping images')
    
    for target_image in files:
        
        img = cv2.imread('images/' + target_image)
        crop_img = img[0:0 + Cheight,560:560 + Cwidth]
        cv2.imwrite('output_images/' + num + target_image, crop_img) 

#------------------------------------------------------#
# Reshaping images (width x height)
#------------------------------------------------------#

if image_reshape:    
    
    print('Reshaping images')
    
    #------------------------------------------------------#
    # Reshaping data for training
    #------------------------------------------------------#

    # loop though all images and annotations
    for target_image in files:
        
        # reading image and converting to numpy array for processing
        img = np.array(cv2.imread('Output_images/' + num + target_image))
    
        old_height, old_width, channels = img.shape
        
        # Resizing image
        output_size = (Rwidth, Rheight)
        Reshape_img = cv2.resize(img, output_size)
        
        cv2.imwrite('Output_images/' + num + target_image, Reshape_img) 
        
    #------------------------------------------------------#
    # Moving bounding box (not need for normilised data)
    #------------------------------------------------------#
        #if bounding:
        
            # incert_XML stuff here (deleted it by accident) 

#------------------------------------------------------#
# Augmenting Data
#------------------------------------------------------#

if data_augmentation:
    
    print('Augmenting data.')
    
    if Rotations: print('Rotations only')
    
    for target_image in files:
        
        # Getting file name to match with annotation
        fileName, ext = os.path.splitext(target_image)
        
        # Opening target image
        originalImage = np.array(cv2.imread('images/' + target_image))
        
        # Where the input annotation txt is file is
        original_txt = input_txt + fileName + '.txt'
        
        # Working out height and with of input image
        height, width = originalImage.shape[:2]
        
        # Opening input .txt file to read old annotations
        file = open(original_txt) 
        
        # Reading annotation content
        content = file.read()
        
        # Makiung a list of the annotations
        params = content.split(" ")
        
        # Close file
        file.close
        
        # Old annotations
        group, a, b, max_a, max_b = params
        
        # Used for writing new annotation
        gap = ' '
        
        if not Rotations:
            #------------------------------------------------------#
            # GrayScale
            #------------------------------------------------------#
            
            # Converting to grayscale
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            
            # Output file with new annotation
            cv2.imwrite('output_images/' + fileName + '_Gray' + '.jpg', grayImage) 
            
            # Location and name of the file to be saved
            saveGray = 'output_annotations/' + fileName + '_Gray' + '.txt'
            
            # Copy annotation from input and save as the grayscale annotation (bounding boxes are the same)
            shutil.copyfile(original_txt, saveGray)
            
            #------------------------------------------------------#
            # Flip
            #------------------------------------------------------#
            
            # 0 means flip round the x axis, 1 is y axis
            flipImage = cv2.flip(originalImage, 1)
            
            cv2.imwrite('output_images/' + fileName + '_Flip' + '.jpg', flipImage)
            
            # Location and name of the file to be saved
            saveFlip = 'output_annotations/' + fileName + '_Flip' + '.txt'
            
            # New annotation values
            #new_a = (1 - float(a))
            
            # Making new annotation (.txt)
            new_txt = open(saveFlip,"w+")
            
            # Writing new annotation values
            #new_txt.write(str(group) + gap + str(new_a) + gap + str(b) + gap + str(max_a) + gap + str(max_b) + gap)
            
            # Close file
            new_txt.close()
        
        #------------------------------------------------------#
        # Rotations (Rot1 and Rot2 defined above) 
        # From: https://github.com/usmanr149/Yolo_bbox_manipulation/blob/master/rotate.py
        #------------------------------------------------------#
        
        elif Rotations:
            
            # initiate the classes
            rot1 = yoloRotatebbox(input_Image + fileName,  ext, Rot1)
            rot12 = yoloRotatebbox(input_Image + fileName, ext, -Rot1)
            rot2 = yoloRotatebbox(input_Image + fileName, ext, Rot2)
            rot22 = yoloRotatebbox(input_Image + fileName, ext, -Rot2)
            
            
            bbox1 = rot1.rotateYolobbox()
            Rot_image1 = rot1.rotate_image()
            
            # to write rotateed image to disk
            cv2.imwrite(fileName + '_' + 'rot_1' + '.jpg', Rot_image1)
        
            file_name = (fileName + '_' + 'rot_1' + '.txt')
            if os.path.exists(file_name):
                os.remove(file_name)
        
            # to write the new rotated bboxes to file
            for i in bbox1:
                with open(file_name, 'a') as fout:
                    fout.writelines(
                        ' '.join(map(str, cvFormattoYolo(i, rot1.rotate_image().shape[0], rot1.rotate_image().shape[1]))) + '\n')
            
            
            bbox12 = rot12.rotateYolobbox()
            Rot_image12 = rot12.rotate_image()
            
            # to write rotateed image to disk
            cv2.imwrite(fileName + '_' + 'rot_1n' + '.jpg', Rot_image12)
        
            file_name = (fileName + '_' + 'rot_1n' + '.txt')
            if os.path.exists(file_name):
                os.remove(file_name)
        
            # to write the new rotated bboxes to file
            for i in bbox12:
                with open(file_name, 'a') as fout:
                    fout.writelines(
                        ' '.join(map(str, cvFormattoYolo(i, rot12.rotate_image().shape[0], rot12.rotate_image().shape[1]))) + '\n')
            
            
            bbox2 = rot2.rotateYolobbox()
            Rot_image2 = rot2.rotate_image()
            
            # to write rotateed image to disk
            cv2.imwrite(fileName + '_' + 'rot_2' + '.jpg', Rot_image2)
        
            file_name = (fileName + '_' + 'rot_2' + '.txt')
            if os.path.exists(file_name):
                os.remove(file_name)
        
            # to write the new rotated bboxes to file
            for i in bbox2:
                with open(file_name, 'a') as fout:
                    fout.writelines(
                        ' '.join(map(str, cvFormattoYolo(i, rot2.rotate_image().shape[0], rot2.rotate_image().shape[1]))) + '\n')
                    
                    
            bbox22 = rot22.rotateYolobbox()
            Rot_image22 = rot22.rotate_image()
            
            # to write rotateed image to disk
            cv2.imwrite(fileName + '_' + 'rot_2n' + '.jpg', Rot_image22)
        
            file_name = (fileName + '_' + 'rot_2n' + '.txt')
            if os.path.exists(file_name):
                os.remove(file_name)
        
            # to write the new rotated bboxes to file
            for i in bbox22:
                with open(file_name, 'a') as fout:
                    fout.writelines(
                        ' '.join(map(str, cvFormattoYolo(i, rot22.rotate_image().shape[0], rot22.rotate_image().shape[1]))) + '\n')
            
        
        
print('Done')


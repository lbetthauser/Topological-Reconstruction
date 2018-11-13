#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=3
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image, ImageFilter
import subprocess
import sys
import timeit


# Creates a Numpy Array representing a full elementary cubical complex
probability_of_insertion = .5
size_of_image = (10,10,10)
voxel_Matrix = np.random.choice([0, 1], size_of_image, p=[1-probability_of_insertion, probability_of_insertion])


#Stores and accesses directory of helper functions, image, and perseus executable
helper_function_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code\Reconstruction_Pipeline_3d"
#image_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code\3-dimensional Grayscale Images"
perseus_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code"


# Checks the accuracy of the restored image (first number is the number of differing pixels, second number is accuracy of restoration)
def check_Accuracy(Matrix_1, Matrix_2):
    number_Total_Pixels = Matrix_1.shape[0]*Matrix_1.shape[1]*Matrix_1.shape[2]
    number_Correct_Assignments = np.sum(Matrix_1==Matrix_2)
    percent_Accurate_Pixels = number_Correct_Assignments/number_Total_Pixels
    print("Reconstruction Accuracy")
    print(number_Total_Pixels-number_Correct_Assignments)
    print(percent_Accurate_Pixels)
    return

# pipeline for reconstructing a grayscale image using Moebius Inversion
os.chdir(helper_function_dir)
import Moebius_Inversion_d3 as mi


# Displays speed of Moebius Inverting a grayscale image
start = timeit.default_timer()
moebius_Inverted_Matrix = mi.Moebius_Invert(voxel_Matrix)
stop = timeit.default_timer()
print("The time it takes to invert the matrix is %f seconds." % (stop-start))


# Reconstructs grayscale image by convolving Moebius_Inverted_Matrix with the Zeta function
start = timeit.default_timer()
restored_Image = mi.convolution_Zeta(moebius_Inverted_Matrix)
stop = timeit.default_timer()
print("The time it takes to invert the matrix is %f seconds." % (stop-start))
# Prints accuracy of reconstruction for User
check_Accuracy(restored_Image, voxel_Matrix)


# imports Euler Characteristic Transform file
os.chdir(helper_function_dir)
import ECT_Voxels_d3 as ect


#Constructs Euler Characteristic Curves 
start = timeit.default_timer()
euler_Characteristic_Curves = ect.Compute_Euler_Characteristic_Curves(voxel_Matrix)
ecc_f0 = euler_Characteristic_Curves[0]
ecc_f1 = euler_Characteristic_Curves[1]
ecc_f2 = euler_Characteristic_Curves[2]
ecc_f3 = euler_Characteristic_Curves[3]
ecc_f4 = euler_Characteristic_Curves[4]
ecc_f5 = euler_Characteristic_Curves[5]
ecc_f6 = euler_Characteristic_Curves[6]
ecc_f7 = euler_Characteristic_Curves[7]
stop = timeit.default_timer()
print("The time it takes to compute Euler Characteristic Curves is %f seconds." % (stop-start))


#constructs the moebius_Inversion_Matrix from Betti Numbers
start = timeit.default_timer()
restored_Moebius_Inverted_Matrix1 = ect.generate_Inversion_Values(voxel_Matrix, ecc_f0, ecc_f1, ecc_f2, ecc_f3, ecc_f4, ecc_f5, ecc_f6, ecc_f7)
stop = timeit.default_timer()
check_Accuracy(restored_Moebius_Inverted_Matrix1, moebius_Inverted_Matrix)
print("The time it takes to construct the moebius_Inverted_Matrix from the Euler Characteristic Curves is %f seconds." % (stop-start))


# reconstructs the image by convolving the restored_Moebius_Inverted_Matrix with the Zeta function
restored_Image1 = ect.convolution_Zeta(restored_Moebius_Inverted_Matrix1)
check_Accuracy(restored_Image1, voxel_Matrix)


# imports visualization file for 3D rendering
os.chdir(helper_function_dir)
import Voxel_Visualization_d3 as vis

# 3D visualization of voxel data
arr = voxel_Matrix
# normalizes and rescales image to improve rendering time
transformed = np.clip(vis.scale_by(np.clip(vis.normalize(arr)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)

# rescales image to user's choice
img_Dim = 15
from skimage.transform import resize
resized = resize(transformed, (img_Dim, img_Dim, img_Dim), mode='constant')
# renders image
vis.plot_cube(resized, img_Dim)
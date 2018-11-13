#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image, ImageFilter
import subprocess
import sys
import timeit




#Stores and accesses directory of helper functions, image, and perseus executable
helper_function_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code\Reconstruction_Pipeline_2d"
image_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code\2-dimensional Grayscale Images"
perseus_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code"
image_name = 'Mona-Lisa.jpg'
#Opens the image directory to construct the pixel matrix
os.chdir(image_dir)
image = PIL.Image.open(image_name, 'r').convert('L')
image.save('greyscale.png')
#image.show()
pic = np.asarray(image, dtype='int')
# renames picture as a variable used for future functions
pixel_Matrix = pic
# records size of image for future functions [for proof of concept this information is known without having access to image]
size_of_image = [pic.shape[0],pic.shape[1]]
plt.imshow(pixel_Matrix, cmap = plt.get_cmap('gray'))
#plt.show()


# Checks the accuracy of the restored image (first number is the number of differing pixels, second number is accuracy of restoration)
def check_Accuracy(Matrix_1, Matrix_2):
    number_Total_Pixels = Matrix_1.shape[0]*Matrix_1.shape[1]
    number_Correct_Assignments = np.sum(Matrix_1==Matrix_2)
    percent_Accurate_Pixels = number_Correct_Assignments/number_Total_Pixels
    print("Reconstruction Accuracy")
    print(number_Total_Pixels-number_Correct_Assignments)
    print(percent_Accurate_Pixels)
    return

# pipeline for reconstructing a grayscale image using Moebius Inversion
os.chdir(helper_function_dir)
import Moebius_Inversion_d2 as mi
import ECT_Voxels_d2 as ect
import PHT_Voxels_d2 as pht


# Displays speed of Moebius Inverting a grayscale image
start = timeit.default_timer()
moebius_Inverted_Matrix = mi.Moebius_Invert(pixel_Matrix)
stop = timeit.default_timer()
print("The time it takes to invert the matrix is %f seconds." % (stop-start))
# Creates Image of Moebius Inverted Matrix
plt.imshow(abs(moebius_Inverted_Matrix), cmap = plt.get_cmap('gray'))
#plt.show()


# Creates a 2 dimensional Support Array for Image from Moebius Inverted Matrix (position, inversion value)
start = timeit.default_timer()
support_Image = mi.Support_Moebius_Inversion(moebius_Inverted_Matrix)
stop = timeit.default_timer()
print("The time it takes to construct the support array is %f seconds." % (stop-start))


# Times reconstruction of Moebius_Inverted_Matrix	
start = timeit.default_timer()
restored_Moebius_Inverted_Matrix = mi.Reconstruct_Moebius_Inverted_Matrix(support_Image, size_of_image)
stop = timeit.default_timer()
print("The time it takes to reconstruct the moebius_Inverted_Matrix is %f seconds." % (stop-start))
# Prints accuracy of reconstruction for User
check_Accuracy(moebius_Inverted_Matrix, restored_Moebius_Inverted_Matrix)


# Displays restored Image as a grayscale image to user
start = timeit.default_timer()
restored_Image = mi.convolution_Zeta(restored_Moebius_Inverted_Matrix)
stop = timeit.default_timer()
print("The time it takes to reconstruct the image is %f seconds." % (stop-start))
# Prints accuracy of reconstruction for User
check_Accuracy(pixel_Matrix, restored_Image)
plt.imshow(restored_Image, cmap = plt.get_cmap('gray'))
#plt.show()


# imports Euler Characteristic Transform file
os.chdir(helper_function_dir)
import ECT_Voxels_d2 as ect


#Constructs Euler Characteristic Curves 
start = timeit.default_timer()
euler_Characteristic_Curves = ect.Compute_Euler_Characteristic_Curves(pixel_Matrix)
ecc_f0 = euler_Characteristic_Curves[0]
ecc_f1 = euler_Characteristic_Curves[1]
ecc_f2 = euler_Characteristic_Curves[2]
ecc_f3 = euler_Characteristic_Curves[3]
stop = timeit.default_timer()
print("The time it takes to compute Euler Characteristic Curves is %f seconds." % (stop-start))


#constructs the moebius_Inversion_Matrix from Betti Numbers
start = timeit.default_timer()
restored_Moebius_Inverted_Matrix1 = ect.generate_Inversion_Values(pixel_Matrix, ecc_f0, ecc_f1, ecc_f2, ecc_f3)
stop = timeit.default_timer()
check_Accuracy(restored_Moebius_Inverted_Matrix1, moebius_Inverted_Matrix)
print("The time it takes to construct the moebius_Inverted_Matrix from the Euler Characteristic Curves is %f seconds." % (stop-start))
restored_Image1 = ect.convolution_Zeta(restored_Moebius_Inverted_Matrix1)
plt.imshow(restored_Image1, cmap = plt.get_cmap('gray'))
#plt.show()


# Currently the PHT functionality of this code only handles binary 2-dimensional images (not grayscale)
# imports Persistenet Homology Transform file
os.chdir(helper_function_dir)
import PHT_Voxels_d2 as pht

# Creates a Numpy Array representing a full elementary cubical complex
probability_of_insertion = .2
pixel_Matrix = np.random.choice([0, 1], size=(10,10), p=[1-probability_of_insertion, probability_of_insertion])
size_of_image = pixel_Matrix.shape
image_name = 'test'
plt.imshow(pixel_Matrix, cmap = plt.get_cmap('gray'))
plt.show()

# Builds perseus input of lower star filtration
start = timeit.default_timer()
pht.Construct_Lower_Dimensional_Faces(image_name, 0, pixel_Matrix)
pht.Construct_Lower_Dimensional_Faces(image_name, 1, pixel_Matrix)
pht.Construct_Lower_Dimensional_Faces(image_name, 2, pixel_Matrix)
pht.Construct_Lower_Dimensional_Faces(image_name, 3, pixel_Matrix)
stop = timeit.default_timer()
print("The time it takes to construct the Perseus array is %f seconds." % (stop-start))


#Constructs the barcodes (will switch working directory to perseus directory to store files) 
start = timeit.default_timer()
pht.Construct_Barcodes(image_name, pixel_Matrix)
stop = timeit.default_timer()
print("The time it takes to construct the Perseus array is %f seconds." % (stop-start))
print("Perseus has finished computing")


#Compute Euler Characteristic Changes using Perseus Output (assumes knowledge of the size of the original image)
euler_Array0 = pht.Euler_Characteristic_Change(image_name, 0, size_of_image)
euler_Array1 = pht.Euler_Characteristic_Change(image_name, 1, size_of_image)
euler_Array2 = pht.Euler_Characteristic_Change(image_name, 2, size_of_image)
euler_Array3 = pht.Euler_Characteristic_Change(image_name, 3, size_of_image)

#constructs the moebius_Inversion_Matrix from Betti Numbers
restored_Moebius_Inverted_Matrix2 = pht.generate_Inversion_Values(pixel_Matrix, euler_Array0, euler_Array1, euler_Array2, euler_Array3)


# Displays image from matrix (reverses the values so that 0 is white and 1 is black)
restored_Image2 = pht.convolution_Zeta(restored_Moebius_Inverted_Matrix2)
plt.imshow(restored_Image2, cmap = plt.get_cmap('gray'))
plt.show()
# Prints accuracy of reconstruction for User
check_Accuracy(pixel_Matrix, restored_Image2)
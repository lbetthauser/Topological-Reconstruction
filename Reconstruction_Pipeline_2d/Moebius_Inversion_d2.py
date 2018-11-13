#Topological Voxel Reconstruction
#Reconstructs Grayscale images for dimension d=2 using Moebius Inversion
import numpy as np

# Moebius Invert Pixels 
def Moebius_Invert(pixel_Matrix):
	size_of_image = pixel_Matrix.shape
	moebius_Inverted_Matrix = np.zeros((size_of_image[0], size_of_image[1]), dtype='int')
	# Initial condition required for Moebius Inversion
	moebius_Inverted_Matrix[0,0] = pixel_Matrix[0,0]
	# Special case for first row of image
	j = 1
	old_entry = pixel_Matrix[0,j-1]
	for j in range(1,size_of_image[1]):
		new_entry = pixel_Matrix[0,j]
		moebius_Inverted_Matrix[0,j] = new_entry - old_entry
		old_entry = new_entry
	# Special case for first column of image
	i = 1
	old_entry = pixel_Matrix[i-1,0]
	for i in range(1,size_of_image[0]):
		new_entry = pixel_Matrix[i,0] 
		moebius_Inverted_Matrix[i,0] = new_entry - old_entry
		old_entry = new_entry
	# General formula for inversion
	for i in range(1,size_of_image[0]):
		j = 1
		left_sum = pixel_Matrix[i,j-1] - pixel_Matrix[i-1,j-1]
		for j in range(1,size_of_image[1]):
			right_sum = pixel_Matrix[i,j] - pixel_Matrix[i-1,j]
			moebius_Inverted_Matrix[i,j] = right_sum - left_sum
			left_sum = right_sum
	return(moebius_Inverted_Matrix)


# Store support of Moebius Inversion
def Support_Moebius_Inversion(moebius_Inverted_Matrix):
	support_Image = np.zeros((moebius_Inverted_Matrix.shape[0]*moebius_Inverted_Matrix.shape[1],2))
	index = 0
	for i in range(moebius_Inverted_Matrix.shape[0]):
		for j in range(moebius_Inverted_Matrix.shape[1]):
			if moebius_Inverted_Matrix[i,j] != 0:
				# stores position using quotients and remainders
				support_Image[index,0] = i*moebius_Inverted_Matrix.shape[1] + j
				# stores inverted value for reconstruction
				support_Image[index,1] = moebius_Inverted_Matrix[i,j]
				index += 1
	# removes entries not corresponding with elements of the support of the Matrix
	support_Image = np.unique(support_Image, axis = 0)
	if support_Image[0,0] == 0:
		support_Image = np.delete(support_Image, 0, 0)
	return(support_Image)


# Reconstructs moebius_Inverted_Matrix from the support of the image
def Reconstruct_Moebius_Inverted_Matrix(support_Image, size_of_image):
	moebius_Inverted_Matrix = np.zeros((size_of_image[0], size_of_image[1]))
	for i in range(support_Image.shape[0]):
		pixel_Image_Info = support_Image[i]
		row = int(pixel_Image_Info[0]//size_of_image[1])
		column = int(pixel_Image_Info[0]%size_of_image[1])
		moebius_Inverted_Matrix[row,column] = pixel_Image_Info[1]
	return(moebius_Inverted_Matrix)


# Convoluting using Moebius Inverted Matrix
def convolution_Zeta(moebius_Inverted_Matrix):
	size_of_image = moebius_Inverted_Matrix.shape
	restored_Image = np.zeros((moebius_Inverted_Matrix.shape[0],moebius_Inverted_Matrix.shape[1]), dtype='int')
	# special case for least element
	restored_Image[0,0] = moebius_Inverted_Matrix[0,0]
	# Special case for first row of image
	j = 1
	W_sum = restored_Image[0,j-1]
	for j in range(1,size_of_image[1]):
		E_sum = moebius_Inverted_Matrix[0,j] + W_sum
		restored_Image[0,j] = E_sum
		W_sum = E_sum
	# Special case for first column of image
	i = 1
	N_sum = restored_Image[i-1,0]
	for i in range(1,size_of_image[0]):
		S_sum = moebius_Inverted_Matrix[i,0] + N_sum
		restored_Image[i,0] = S_sum
		N_sum = S_sum
	# General case for convolution
	for i in range(1,restored_Image.shape[0]):
		j = 1
		NW_sum = restored_Image[i-1,j-1]
		SW_sum = restored_Image[i,j-1]
		for j in range(1,restored_Image.shape[1]):
			# stores inversion value for pixel locally
			NE_sum = restored_Image[i-1,j]
			# computes convolution using inclusion-exclusion
			SE_sum = moebius_Inverted_Matrix[i,j] + NE_sum + SW_sum - NW_sum
			restored_Image[i,j] = SE_sum
			NW_sum = NE_sum
			SW_sum = SE_sum
	return(restored_Image)
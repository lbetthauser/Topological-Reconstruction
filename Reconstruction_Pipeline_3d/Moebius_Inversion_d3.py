#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=2
import numpy as np


# Moebius Invert Pixels 
def Moebius_Invert(voxel_Matrix):
	size_of_image = [voxel_Matrix.shape[0], voxel_Matrix.shape[1], voxel_Matrix.shape[2]]
	moebius_Inverted_Matrix = np.zeros((size_of_image[0], size_of_image[1], size_of_image[2]), dtype='int')
	# Initial condition required for Moebius Inversion
	moebius_Inverted_Matrix[0,0,0] = voxel_Matrix[0,0,0]
	# Special case for first column of YZ plane
	i = 1
	old_entry = voxel_Matrix[i-1,0,0]
	for i in range(1,size_of_image[0]):
		new_entry = voxel_Matrix[i,0,0]
		moebius_Inverted_Matrix[i,0,0] = new_entry - old_entry
		old_entry = new_entry
	# Special case for first column of XZ plane
	j = 1
	old_entry = voxel_Matrix[0,j-1,0]
	for j in range(1,size_of_image[1]):
		new_entry = voxel_Matrix[0,j,0]
		moebius_Inverted_Matrix[0,j,0] = new_entry - old_entry
		old_entry = new_entry
	# Special case for first column of XY plane
	k = 1
	old_entry = voxel_Matrix[0,0,k-1]
	for k in range(1,size_of_image[2]):
		new_entry = voxel_Matrix[0,0,k]
		moebius_Inverted_Matrix[0,0,k] = new_entry - old_entry
		old_entry = new_entry
	# 2 dimensional case to invert the YZ plane 
	for j in range(1,size_of_image[1]):
		k = 1
		left_sum = voxel_Matrix[0,j,k-1] - voxel_Matrix[0,j-1,k-1]
		for k in range(1,size_of_image[2]):
			right_sum = voxel_Matrix[0,j,k] - voxel_Matrix[0,j-1,k]
			moebius_Inverted_Matrix[0,j,k] = right_sum - left_sum
			left_sum = right_sum
	# 2 dimensional case to invert the XZ plane
	for i in range(1,size_of_image[0]):
		k = 1
		left_sum = voxel_Matrix[i,0,k-1] - voxel_Matrix[i-1,0,k-1]
		for k in range(1,size_of_image[2]):
			right_sum = voxel_Matrix[i,0,k] - voxel_Matrix[i-1,0,k]
			moebius_Inverted_Matrix[i,0,k] = right_sum - left_sum
			left_sum = right_sum
	# 2 dimensional case to invert the XY plane
	for i in range(1,size_of_image[0]):
		j = 1
		left_sum = voxel_Matrix[i,j-1,0] - voxel_Matrix[i-1,j-1,0]
		for j in range(1,size_of_image[1]):
			right_sum = voxel_Matrix[i,j,0] - voxel_Matrix[i-1,j,0]
			moebius_Inverted_Matrix[i,j,0] = right_sum - left_sum
			left_sum = right_sum
	# General formula for inversion
	for i in range(1,size_of_image[0]):
		for j in range(1,size_of_image[1]):
			k = 1
			back_NW = voxel_Matrix[i-1,j-1,k-1]
			back_SW = voxel_Matrix[i-1,j,k-1]
			back_left_sum = back_SW - back_NW
			front_NW = voxel_Matrix[i,j-1,k-1]
			front_SW = voxel_Matrix[i,j,k-1]
			front_left_sum = front_NW - front_SW
			left_sum = front_left_sum + back_left_sum
			for k in range(1,size_of_image[2]):
				back_NE = voxel_Matrix[i-1,j-1,k]
				back_SE = voxel_Matrix[i-1,j,k]
				front_NE = voxel_Matrix[i,j-1,k]
				front_SE = voxel_Matrix[i,j,k] 
				right_sum = front_SE - front_NE - back_SE + back_NE
				moebius_Inverted_Matrix[i,j,k] = right_sum + left_sum
				# when updating we must multiply by (-1)^(dimension)
				left_sum = -right_sum
	return(moebius_Inverted_Matrix)


# Convoluting using Moebius Inverted Matrix
def convolution_Zeta(moebius_Inverted_Matrix):
	size_of_image = moebius_Inverted_Matrix.shape
	restored_Image = np.zeros((moebius_Inverted_Matrix.shape[0],moebius_Inverted_Matrix.shape[1], moebius_Inverted_Matrix.shape[2]), dtype='int')
	# special case for least element
	restored_Image[0,0,0] = moebius_Inverted_Matrix[0,0,0]
	# Special case for first column of YZ plane
	i = 1
	previous_sum = moebius_Inverted_Matrix[i-1,0,0]
	for i in range(1,size_of_image[0]):
		new_entry = moebius_Inverted_Matrix[i,0,0]
		new_sum = new_entry + previous_sum
		restored_Image[i,0,0] = new_sum
		previous_sum = new_sum
	# Special case for first column of XZ plane
	j = 1
	previous_sum = moebius_Inverted_Matrix[0,j-1,0]
	for j in range(1,size_of_image[1]):
		new_entry = moebius_Inverted_Matrix[0,j,0]
		new_sum = new_entry + previous_sum
		restored_Image[0,j,0] = new_sum
		previous_sum = new_sum
	# Special case for first column of XY plane
	k = 1
	previous_sum = moebius_Inverted_Matrix[0,0,k-1]
	for k in range(1,size_of_image[2]):
		new_entry = moebius_Inverted_Matrix[0,0,k]
		new_sum = new_entry + previous_sum
		restored_Image[0,0,k] = new_sum
		previous_sum = new_sum
	# 2 dimensional case to invert the YZ plane 
	for j in range(1,restored_Image.shape[1]):
		k = 1
		NW_sum = restored_Image[0,j-1,k-1]
		SW_sum = restored_Image[0,j,k-1]
		for k in range(1,restored_Image.shape[2]):
			# stores inversion value for pixel locally
			NE_sum = restored_Image[0,j-1,k]
			# computes convolution using inclusion-exclusion
			SE_sum = moebius_Inverted_Matrix[0,j,k] + NE_sum + SW_sum - NW_sum
			restored_Image[0,j,k] = SE_sum
			NW_sum = NE_sum
			SW_sum = SE_sum
	# 2 dimensional case to invert the XZ plane 
	for i in range(1,restored_Image.shape[0]):
		k = 1
		NW_sum = restored_Image[i-1,0,k-1]
		SW_sum = restored_Image[i,0,k-1]
		for k in range(1,restored_Image.shape[2]):
			# stores inversion value for pixel locally
			NE_sum = restored_Image[i-1,0,k]
			# computes convolution using inclusion-exclusion
			SE_sum = moebius_Inverted_Matrix[i,0,k] + NE_sum + SW_sum - NW_sum
			restored_Image[i,0,k] = SE_sum
			NW_sum = NE_sum
			SW_sum = SE_sum
	# 2 dimensional case to invert the XY plane 
	for i in range(1,restored_Image.shape[0]):
		j = 1
		NW_sum = restored_Image[i-1,j-1,0]
		SW_sum = restored_Image[i,j-1,0]
		for j in range(1,restored_Image.shape[1]):
			# stores inversion value for pixel locally
			NE_sum = restored_Image[i-1,j,0]
			# computes convolution using inclusion-exclusion
			SE_sum = moebius_Inverted_Matrix[i,j,0] + NE_sum + SW_sum - NW_sum
			restored_Image[i,j,0] = SE_sum
			NW_sum = NE_sum
			SW_sum = SE_sum
	# General formula for inversion using inclusion-exclusion
	for i in range(1,size_of_image[0]):
		for j in range(1,size_of_image[1]):
			k = 1
			back_NW = restored_Image[i-1,j-1,k-1]
			back_SW = restored_Image[i-1,j,k-1]
			front_NW = restored_Image[i,j-1,k-1]
			front_SW = restored_Image[i,j,k-1]
			for k in range(1,size_of_image[2]):
				back_NE = restored_Image[i-1,j-1,k]
				back_SE = restored_Image[i-1,j,k]
				front_NE = restored_Image[i,j-1,k]
				front_SE = moebius_Inverted_Matrix[i,j,k] + front_NE + front_SW + back_SE - front_NW - back_NE - back_SW + back_NW
				restored_Image[i,j,k] = front_SE
				# update moving window with known values
				back_NW = back_NE
				back_SW = back_SE
				front_NW = front_NE
				front_SW = front_SE
	return(restored_Image)
#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=2
import numpy as np


# Compute Euler Characteristic Curve from full cubes 
def Compute_Euler_Characteristic_Curves(pixel_Matrix):
	# indexes for number of critical vertices for respective ECC
	number_support_vertices_f0 = 0
	number_support_vertices_f1 = 0
	number_support_vertices_f2 = 0
	number_support_vertices_f3 = 0
	# stores [filtration value, nonzero Euler Characteristic]
	Euler_Characteristic_Curve_f0 = np.zeros((4*pixel_Matrix.shape[0]*pixel_Matrix.shape[1],2))
	Euler_Characteristic_Curve_f1 = np.zeros((4*pixel_Matrix.shape[0]*pixel_Matrix.shape[1],2))
	Euler_Characteristic_Curve_f2 = np.zeros((4*pixel_Matrix.shape[0]*pixel_Matrix.shape[1],2))
	Euler_Characteristic_Curve_f3 = np.zeros((4*pixel_Matrix.shape[0]*pixel_Matrix.shape[1],2))
	# special cases
	# corners in order of filtrations (NW, NE, SE, SW)
	if pixel_Matrix[0,0] != 0:
		Euler_Characteristic_Curve_f0[0,1] = pixel_Matrix[0,0]
		number_support_vertices_f0 += 1
	if pixel_Matrix[0,pixel_Matrix.shape[1]-1] != 0:
		Euler_Characteristic_Curve_f1[0,1] = pixel_Matrix[0,pixel_Matrix.shape[1]-1]
		number_support_vertices_f1 += 1
	if pixel_Matrix[pixel_Matrix.shape[0]-1,pixel_Matrix.shape[1]-1] != 0:
		Euler_Characteristic_Curve_f2[0,1] = pixel_Matrix[pixel_Matrix.shape[0]-1,pixel_Matrix.shape[1]-1]
		number_support_vertices_f2 += 1
	if pixel_Matrix[pixel_Matrix.shape[0]-1,0] != 0:
		Euler_Characteristic_Curve_f3[0,1] = pixel_Matrix[pixel_Matrix.shape[0]-1,0]
		number_support_vertices_f3 += 1
	# cube indexes are in order of f0 filtration (cube0 = NW, cube1 = SW, cube2 = NE, cube3= SE)
	# left boundary
	column = 0
	cube0 = pixel_Matrix[0,column]
	for row in range(1,pixel_Matrix.shape[0]):
		cube1 = pixel_Matrix[row,column]
		card_vertex = max(cube0, cube1)
		card_edge_f0 = cube0
		card_edge_f3 = cube1
		euler_characteristic_f0 = card_vertex - card_edge_f0
		euler_characteristic_f3 = card_vertex - card_edge_f3
		if euler_characteristic_f0 != 0:
			cube_value = (pixel_Matrix.shape[0]*column+row)+1
			vertex_filtration_value = cube_value + column
			Euler_Characteristic_Curve_f0[number_support_vertices_f0,0] = vertex_filtration_value - 1
			Euler_Characteristic_Curve_f0[number_support_vertices_f0,1] = euler_characteristic_f0
			number_support_vertices_f0 += 1
		if euler_characteristic_f3 != 0:
			cube_value = ((pixel_Matrix.shape[0]-row-1)*pixel_Matrix.shape[1]+column)+1
			vertex_filtration_value = cube_value + (pixel_Matrix.shape[0]-1 - row)
			Euler_Characteristic_Curve_f3[number_support_vertices_f3,0] = vertex_filtration_value - 1
			Euler_Characteristic_Curve_f3[number_support_vertices_f3,1] = euler_characteristic_f3
			number_support_vertices_f3 += 1
		# updates sliding window with known entries
		cube0 = cube1
	# top boundary
	row = 0
	cube0 = pixel_Matrix[row,0]
	for column in range(1,pixel_Matrix.shape[1]):
		cube2 = pixel_Matrix[row,column]
		card_vertex = max(cube0, cube2)
		card_edge_f0 = cube0
		card_edge_f1 = cube2
		euler_characteristic_f0 = card_vertex - card_edge_f0
		euler_characteristic_f1 = card_vertex - card_edge_f1		
		if euler_characteristic_f0 != 0:
			cube_value = (pixel_Matrix.shape[0]*column+row)+1
			vertex_filtration_value = cube_value + column
			Euler_Characteristic_Curve_f0[number_support_vertices_f0,0] = vertex_filtration_value - 1
			Euler_Characteristic_Curve_f0[number_support_vertices_f0,1] = euler_characteristic_f0
			number_support_vertices_f0 += 1
		if euler_characteristic_f1 != 0:
			cube_value = (pixel_Matrix.shape[1]*row+(pixel_Matrix.shape[1]-column-1))+1
			vertex_filtration_value = cube_value + row
			Euler_Characteristic_Curve_f1[number_support_vertices_f1,0] = vertex_filtration_value - 1
			Euler_Characteristic_Curve_f1[number_support_vertices_f1,1] = euler_characteristic_f1
			number_support_vertices_f1 += 1
		# updates sliding window with known entries
		cube0 = cube2
	# general case
	for column in range(1,pixel_Matrix.shape[1]):
		row = 1
		cube0 = pixel_Matrix[row-1,column-1]
		cube2 = pixel_Matrix[row-1,column]
		for row in range(1,pixel_Matrix.shape[0]):
			cube1 = pixel_Matrix[row,column-1]
			cube3 = pixel_Matrix[row,column]
			card_vertex = max(cube0, cube1, cube2, cube3)
			card_edge0_f0 = max(cube0, cube1)
			card_edge1_f0 = max(cube0, cube2)
			card_square_f0 = cube0
			card_edge0_f1 = card_edge1_f0
			card_edge1_f1 = max(cube2, cube3)
			card_square_f1 = cube2
			card_edge0_f2 = card_edge1_f1
			card_edge1_f2 = max(cube1, cube3)
			card_square_f2 = cube3
			card_edge0_f3 = card_edge1_f2
			card_edge1_f3 = card_edge0_f0
			card_square_f3 = cube1
			euler_characteristic_f0 = card_vertex - card_edge0_f0 - card_edge1_f0 + card_square_f0
			euler_characteristic_f1 = card_vertex - card_edge0_f1 - card_edge1_f1 + card_square_f1
			euler_characteristic_f2 = card_vertex - card_edge0_f2 - card_edge1_f2 + card_square_f2
			euler_characteristic_f3 = card_vertex - card_edge0_f3 - card_edge1_f3 + card_square_f3
			if euler_characteristic_f0 != 0:
				cube_value = (pixel_Matrix.shape[0]*column+row)+1
				vertex_filtration_value = cube_value + column
				Euler_Characteristic_Curve_f0[number_support_vertices_f0,0] = vertex_filtration_value - 1
				Euler_Characteristic_Curve_f0[number_support_vertices_f0,1] = euler_characteristic_f0
				number_support_vertices_f0 += 1
			if euler_characteristic_f1 != 0:
				cube_value = (pixel_Matrix.shape[1]*row+(pixel_Matrix.shape[1]-column-1))+1
				vertex_filtration_value = cube_value + row
				Euler_Characteristic_Curve_f1[number_support_vertices_f1,0] = vertex_filtration_value - 1
				Euler_Characteristic_Curve_f1[number_support_vertices_f1,1] = euler_characteristic_f1
				number_support_vertices_f1 += 1
			if euler_characteristic_f2 != 0:
				cube_value = ((pixel_Matrix.shape[0]-row-1)+pixel_Matrix.shape[0]*(pixel_Matrix.shape[1]-column-1))+1
				vertex_filtration_value = cube_value + (pixel_Matrix.shape[1]-1 - column)
				Euler_Characteristic_Curve_f2[number_support_vertices_f2,0] = vertex_filtration_value - 1
				Euler_Characteristic_Curve_f2[number_support_vertices_f2,1] = euler_characteristic_f2
				number_support_vertices_f2 += 1
			if euler_characteristic_f3 != 0:
				cube_value = ((pixel_Matrix.shape[0]-row-1)*pixel_Matrix.shape[1]+column)+1
				vertex_filtration_value = cube_value + (pixel_Matrix.shape[0]-1 - row)
				Euler_Characteristic_Curve_f3[number_support_vertices_f3,0] = vertex_filtration_value - 1
				Euler_Characteristic_Curve_f3[number_support_vertices_f3,1] = euler_characteristic_f3
				number_support_vertices_f3 += 1
			# updates sliding window with known entries
			cube0 = cube1
			cube2 = cube3
	# tirms to support of Euler_Characteristic_Curve
	Euler_Characteristic_Curve_f0 = Euler_Characteristic_Curve_f0[:number_support_vertices_f0]
	Euler_Characteristic_Curve_f1 = Euler_Characteristic_Curve_f1[:number_support_vertices_f1]
	Euler_Characteristic_Curve_f2 = Euler_Characteristic_Curve_f2[:number_support_vertices_f2]
	Euler_Characteristic_Curve_f3 = Euler_Characteristic_Curve_f3[:number_support_vertices_f3]
	return(Euler_Characteristic_Curve_f0, Euler_Characteristic_Curve_f1, Euler_Characteristic_Curve_f2, Euler_Characteristic_Curve_f3)


# Constructs the moebius_Inversion_Matrix from Euler Characteristic Curves
def generate_Inversion_Values(pixel_Matrix, euler_Array0, euler_Array1, euler_Array2, euler_Array3):
	size_of_image = pixel_Matrix.shape
	# Allows for placement into extra row and column to account for possible non-zero values which is later trimmed
	moebius_Inversion_Array = np.zeros((size_of_image[0]+1, size_of_image[1]+1), dtype='int')
	# Takes alternating sum of lower star Euler Characteristic for appropriate pixel
	for information_pair in euler_Array0:
		# vertex filtration value
		filtration = 0
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		column = int(position // (size_of_image[0]+1))
		row = int(position  % (size_of_image[0]+1))
		moebius_Inversion_Array[row,column]+= (((-1)**filtration)*inversion_value)
	for information_pair in euler_Array1[1:]:
		# vertex filtration value
		filtration = 1
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		row = int(position // (size_of_image[1]+1))
		column = int(size_of_image[1]-1 - position % (size_of_image[1]+1))
		moebius_Inversion_Array[row,column] += (((-1)**filtration)*inversion_value)
	for information_pair in euler_Array2[1:]:
		# vertex filtration value
		filtration = 2
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		column = int(size_of_image[1]-1 - position // (size_of_image[0]+1))
		row = int(size_of_image[0]-1 - position % (size_of_image[0]+1))
		moebius_Inversion_Array[row,column] += (((-1)**filtration)*inversion_value)
	for information_pair in euler_Array3[1:]:
		# vertex filtration 
		filtration = 3
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		row = int(size_of_image[0]-1 - position // (size_of_image[1]+1))
		column = int(position % (size_of_image[1]+1))
		moebius_Inversion_Array[row,column] += (((-1)**filtration)*inversion_value)
	moebius_Inverted_Matrix = moebius_Inversion_Array[:size_of_image[0],:size_of_image[1]]
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
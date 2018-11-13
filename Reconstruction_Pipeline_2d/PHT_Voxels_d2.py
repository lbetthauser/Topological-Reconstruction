#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=2
import os
import subprocess
import numpy as np


# directory storing perseus executable used to store perseus input and compute barcodes
helper_function_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code\Reconstruction_Pipeline_2d"
perseus_dir = r"C:\Users\leobe\OneDrive\Documents\TDA Code"

'''
Currently only works for binary 2d images (need to integrate different cardinalities for the Construct_Lower_Dimensional_Faces)
'''
# Converts a filtered bitmap into an Array cooresponding with a lower star filtration w.r.t. first quadrant compatible with Perseus (Vidit Nanda software) to compute Persistent Homology
def Construct_Lower_Dimensional_Faces(filename, filtration, pixel_Matrix):
	# creates storage structures for different dimensional faces
	vertex_List = [0]*int(4*pixel_Matrix.shape[0]*pixel_Matrix.shape[1])
	edge_List = [0]*int(4*pixel_Matrix.shape[0]*pixel_Matrix.shape[1])
	square_List = [0]*int(pixel_Matrix.shape[0]*pixel_Matrix.shape[1])
	current_cube_index = 0
	current_edge_index = 0
	current_vertex_index = 0
	number_on_pixels = 0
	vertex_dictionary = {}
	edge_dictionary = {}
	square_dictionary = {}
	if filtration == 0:
		for column in range(pixel_Matrix.shape[1]):
			temp_vertex_dictionary = {}
			temp_edge_dictionary = {}
			temp_square_dictionary = {}
			for row in range(pixel_Matrix.shape[0]):
				if pixel_Matrix[row,column] == 1:
					temp_square_dictionary[current_cube_index] = [current_cube_index]
					square_dictionary[current_cube_index] = [current_cube_index]
					cube_value = (pixel_Matrix.shape[0]*column+row)+1
					vertex_0 = cube_value + column
					vertex_1 = cube_value + column + 1
					vertex_2 = cube_value + int(pixel_Matrix.shape[0]+1) + column
					vertex_3 = cube_value + int(pixel_Matrix.shape[0]+1) + column + 1
					# adds unique vertices to vertex array
					filtration_value = vertex_3
					temp_vertex_array = [vertex_0, vertex_1, vertex_2, vertex_3]
					for vertex_value in temp_vertex_array:
						# stores vertices using filtration value
						if vertex_value not in vertex_dictionary:
							vertex_List[current_vertex_index] = vertex_value
							# adds vertex to current vertices to check
							vertex_dictionary[vertex_value] = current_vertex_index
							# will replace vertex_dictionary when column is changed
							temp_vertex_dictionary[vertex_value] = current_vertex_index
							current_vertex_index += 1
					# identifies edges using indices of the vertices using negatives to indicate orientation
					edge_0 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_1]]
					edge_1 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_2]]
					edge_2 = [vertex_dictionary[vertex_1],-(vertex_dictionary[vertex_3])]
					edge_3 = [vertex_dictionary[vertex_2],-(vertex_dictionary[vertex_3])]
					# stores edges by filtraiton value and checks for redundent entries
					# checks uniqueness of edge for cubes which share edges# boundary of image case
					if vertex_1 in edge_dictionary:
						if len(edge_dictionary[vertex_1]) == 1:
							edge_List[current_edge_index] = [vertex_1, edge_0]
							edge_dictionary[vertex_1].append(current_edge_index)
							current_edge_index += 1
					if vertex_1 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_1, edge_0]
						edge_dictionary[vertex_1] = [current_edge_index]
						current_edge_index += 1
					if vertex_2 in edge_dictionary:
						if len(edge_dictionary[vertex_2]) == 1:
							edge_List[current_edge_index] = [vertex_2, edge_1]
							edge_dictionary[vertex_2].append(current_edge_index)
					if vertex_2 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_2, edge_1]
						edge_dictionary[vertex_2] = [current_edge_index]
						current_edge_index += 1
					# creates entry for back two edges
					edge_List[current_edge_index] = [filtration_value, edge_2]
					edge_dictionary[vertex_3] = [current_edge_index]
					temp_edge_dictionary[vertex_3] = [current_edge_index]
					edge_List[current_edge_index+1] = [filtration_value, edge_3]
					edge_dictionary[vertex_3].append(current_edge_index + 1)
					temp_edge_dictionary[vertex_3].append(current_edge_index)
					current_edge_index += 2
					# creates entry for square
					# edge 0 and 1 are the last entry in the list of edge indices
					edge_0 = edge_dictionary[vertex_1][len(edge_dictionary[vertex_1])-1]
					edge_1 = edge_dictionary[vertex_2][len(edge_dictionary[vertex_2])-1]
					edge_2 = edge_dictionary[vertex_3][0]
					edge_3 = edge_dictionary[vertex_3][1]
					square_List[number_on_pixels] = [filtration_value, edge_0, edge_1, edge_2, edge_3]
					number_on_pixels += 1
					current_cube_index += 1
				else:
					temp_square_dictionary[current_cube_index] = -1
					square_dictionary[current_cube_index] = -1
					current_cube_index += 1
			# clears the dictionaries before moving on to next row to reduce memory
			vertex_dictionary = temp_vertex_dictionary
			#print(vertex_dictionary)
			edge_dictionary = temp_edge_dictionary
			square_dictionary = temp_square_dictionary
			#print(square_dictionary)
		# Removes the 0 tuples used to initiate edge and square lists
		vertex_List = vertex_List[:current_vertex_index]
		edge_List = edge_List[:current_edge_index]
		square_List = square_List[:number_on_pixels]
	if filtration == 1:
		for row in range(pixel_Matrix.shape[0]):
			temp_vertex_dictionary = {}
			temp_edge_dictionary = {}
			temp_square_dictionary = {}
			for column in reversed(range(pixel_Matrix.shape[1])):
				if pixel_Matrix[row,column] == 1:
					temp_square_dictionary[current_cube_index] = [current_cube_index]
					square_dictionary[current_cube_index] = [current_cube_index]
					cube_value = (pixel_Matrix.shape[1]*row+(pixel_Matrix.shape[1]-column-1))+1
					vertex_0 = cube_value + row
					vertex_1 = cube_value + row + 1
					vertex_2 = cube_value + row + int(pixel_Matrix.shape[1]+1)
					vertex_3 = cube_value + row + 1 + int(pixel_Matrix.shape[1]+1)
					# adds unique vertices to vertex array
					filtration_value = vertex_3
					temp_vertex_array = [vertex_0, vertex_1, vertex_2, vertex_3]
					for vertex_value in temp_vertex_array:
						# stores vertices using filtration value
						if vertex_value not in vertex_dictionary:
							vertex_List[current_vertex_index] = vertex_value
							# adds vertex to current vertices to check
							vertex_dictionary[vertex_value] = current_vertex_index
							# will replace vertex_dictionary when column is changed
							temp_vertex_dictionary[vertex_value] = current_vertex_index
							current_vertex_index += 1
					# identifies edges using indices of the vertices
					edge_0 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_1]]
					edge_1 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_2]]
					edge_2 = [vertex_dictionary[vertex_1],-(vertex_dictionary[vertex_3])]
					edge_3 = [vertex_dictionary[vertex_2],-(vertex_dictionary[vertex_3])]
					# stores edges by filtraiton value and checks for redundent entries
					# checks uniqueness of edge for cubes which share edges# boundary of image case
					if vertex_1 in edge_dictionary:
						if len(edge_dictionary[vertex_1]) == 1:
							edge_List[current_edge_index] = [vertex_1, edge_0]
							edge_dictionary[vertex_1].append(current_edge_index)
							current_edge_index += 1
					if vertex_1 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_1, edge_0]
						edge_dictionary[vertex_1] = [current_edge_index]
						current_edge_index += 1
					if vertex_2 in edge_dictionary:
						if len(edge_dictionary[vertex_2]) == 1:
							edge_List[current_edge_index] = [vertex_2, edge_1]
							edge_dictionary[vertex_2].append(current_edge_index)
					if vertex_2 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_2, edge_1]
						edge_dictionary[vertex_2] = [current_edge_index]
						current_edge_index += 1
					# creates entry for back two edges
					edge_List[current_edge_index] = [filtration_value, edge_2]
					edge_dictionary[vertex_3] = [current_edge_index]
					temp_edge_dictionary[vertex_3] = [current_edge_index]
					edge_List[current_edge_index+1] = [filtration_value, edge_3]
					edge_dictionary[vertex_3].append(current_edge_index + 1)
					temp_edge_dictionary[vertex_3].append(current_edge_index)
					current_edge_index += 2
					# creates entry for square
					# edge 0 and 1 are the last entry in the list of edge indices
					edge_0 = edge_dictionary[vertex_1][len(edge_dictionary[vertex_1])-1]
					edge_1 = edge_dictionary[vertex_2][len(edge_dictionary[vertex_2])-1]
					edge_2 = edge_dictionary[vertex_3][0]
					edge_3 = edge_dictionary[vertex_3][1]
					square_List[number_on_pixels] = [filtration_value, edge_0, edge_1, edge_2, edge_3]
					number_on_pixels += 1
					current_cube_index += 1
				else:
					temp_square_dictionary[current_cube_index] = -1
					square_dictionary[current_cube_index] = -1
					current_cube_index += 1
			# clears the dictionaries before moving on to next row to reduce memory
			vertex_dictionary = temp_vertex_dictionary
			edge_dictionary = temp_edge_dictionary
			square_dictionary = temp_square_dictionary
		# Removes the 0 tuples used to initiate edge and square lists
		vertex_List = vertex_List[:current_vertex_index]
		edge_List = edge_List[:current_edge_index]
		square_List = square_List[:number_on_pixels]
	if filtration == 2:
		for column in reversed(range(pixel_Matrix.shape[1])):
			temp_vertex_dictionary = {}
			temp_edge_dictionary = {}
			temp_square_dictionary = {}
			for row in reversed(range(pixel_Matrix.shape[0])):
				if pixel_Matrix[row,column] == 1:
					temp_square_dictionary[current_cube_index] = [current_cube_index]
					square_dictionary[current_cube_index] = [current_cube_index]
					cube_value = ((pixel_Matrix.shape[0]-row-1)+pixel_Matrix.shape[0]*(pixel_Matrix.shape[1]-column-1))+1
					vertex_0 = cube_value + (pixel_Matrix.shape[1]-1 - column)
					vertex_1 = cube_value + (pixel_Matrix.shape[1]-1 - column) + 1
					vertex_2 = cube_value + (pixel_Matrix.shape[1]-1 - column) + int(pixel_Matrix.shape[0]+1)
					vertex_3 = cube_value + (pixel_Matrix.shape[1]-1 - column) + 1 + int(pixel_Matrix.shape[0]+1)
					# adds unique vertices to vertex array
					filtration_value = vertex_3
					temp_vertex_array = [vertex_0, vertex_1, vertex_2, vertex_3]
					for vertex_value in temp_vertex_array:
						# stores vertices using filtration value
						if vertex_value not in vertex_dictionary:
							vertex_List[current_vertex_index] = vertex_value
							# adds vertex to current vertices to check
							vertex_dictionary[vertex_value] = current_vertex_index
							# will replace vertex_dictionary when column is changed
							temp_vertex_dictionary[vertex_value] = current_vertex_index
							current_vertex_index += 1
					# identifies edges using indices of the vertices
					edge_0 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_1]]
					edge_1 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_2]]
					edge_2 = [vertex_dictionary[vertex_1],-(vertex_dictionary[vertex_3])]
					edge_3 = [vertex_dictionary[vertex_2],-(vertex_dictionary[vertex_3])]
					# stores edges by filtraiton value and checks for redundent entries
					# checks uniqueness of edge for cubes which share edges# boundary of image case
					if vertex_1 in edge_dictionary:
						if len(edge_dictionary[vertex_1]) == 1:
							edge_List[current_edge_index] = [vertex_1, edge_0]
							edge_dictionary[vertex_1].append(current_edge_index)
							current_edge_index += 1
					if vertex_1 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_1, edge_0]
						edge_dictionary[vertex_1] = [current_edge_index]
						current_edge_index += 1
					if vertex_2 in edge_dictionary:
						if len(edge_dictionary[vertex_2]) == 1:
							edge_List[current_edge_index] = [vertex_2, edge_1]
							edge_dictionary[vertex_2].append(current_edge_index)
					if vertex_2 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_2, edge_1]
						edge_dictionary[vertex_2] = [current_edge_index]
						current_edge_index += 1
					# creates entry for back two edges
					edge_List[current_edge_index] = [filtration_value, edge_2]
					edge_dictionary[vertex_3] = [current_edge_index]
					temp_edge_dictionary[vertex_3] = [current_edge_index]
					edge_List[current_edge_index+1] = [filtration_value, edge_3]
					edge_dictionary[vertex_3].append(current_edge_index + 1)
					temp_edge_dictionary[vertex_3].append(current_edge_index)
					current_edge_index += 2
					# creates entry for square
					# edge 0 and 1 are the last entry in the list of edge indices
					edge_0 = edge_dictionary[vertex_1][len(edge_dictionary[vertex_1])-1]
					edge_1 = edge_dictionary[vertex_2][len(edge_dictionary[vertex_2])-1]
					edge_2 = edge_dictionary[vertex_3][0]
					edge_3 = edge_dictionary[vertex_3][1]
					square_List[number_on_pixels] = [filtration_value, edge_0, edge_1, edge_2, edge_3]
					number_on_pixels += 1
					current_cube_index += 1
				else:
					temp_square_dictionary[current_cube_index] = -1
					square_dictionary[current_cube_index] = -1
					current_cube_index += 1
			# clears the dictionaries before moving on to next row to reduce memory
			vertex_dictionary = temp_vertex_dictionary
			edge_dictionary = temp_edge_dictionary
			square_dictionary = temp_square_dictionary
		# Removes the 0 tuples used to initiate edge and square lists
		vertex_List = vertex_List[:current_vertex_index]
		edge_List = edge_List[:current_edge_index]
		square_List = square_List[:number_on_pixels]
	if filtration == 3:
		for row in reversed(range(pixel_Matrix.shape[0])):
			temp_vertex_dictionary = {}
			temp_edge_dictionary = {}
			temp_square_dictionary = {}
			for column in (range(pixel_Matrix.shape[1])):
				if pixel_Matrix[row,column] == 1:
					temp_square_dictionary[current_cube_index] = [current_cube_index]
					square_dictionary[current_cube_index] = [current_cube_index]
					cube_value = ((pixel_Matrix.shape[0]-row-1)*pixel_Matrix.shape[1]+column)+1
					vertex_0 = cube_value + (pixel_Matrix.shape[0]-1 - row)
					vertex_1 = cube_value + (pixel_Matrix.shape[0]-1 - row) + 1
					vertex_2 = cube_value + (pixel_Matrix.shape[0]-1 - row) + int(pixel_Matrix.shape[1]+1)
					vertex_3 = cube_value + (pixel_Matrix.shape[0]-1 - row) + 1 + int(pixel_Matrix.shape[1]+1)
					# adds unique vertices to vertex array
					filtration_value = vertex_3
					temp_vertex_array = [vertex_0, vertex_1, vertex_2, vertex_3]
					for vertex_value in temp_vertex_array:
						# stores vertices using filtration value
						if vertex_value not in vertex_dictionary:
							vertex_List[current_vertex_index] = vertex_value
							# adds vertex to current vertices to check
							vertex_dictionary[vertex_value] = current_vertex_index
							# will replace vertex_dictionary when column is changed
							temp_vertex_dictionary[vertex_value] = current_vertex_index
							current_vertex_index += 1
					# identifies edges using indices of the vertices
					edge_0 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_1]]
					edge_1 = [-(vertex_dictionary[vertex_0]),vertex_dictionary[vertex_2]]
					edge_2 = [vertex_dictionary[vertex_1],-(vertex_dictionary[vertex_3])]
					edge_3 = [vertex_dictionary[vertex_2],-(vertex_dictionary[vertex_3])]
					# stores edges by filtraiton value and checks for redundent entries
					# checks uniqueness of edge for cubes which share edges# boundary of image case
					if vertex_1 in edge_dictionary:
						if len(edge_dictionary[vertex_1]) == 1:
							edge_List[current_edge_index] = [vertex_1, edge_0]
							edge_dictionary[vertex_1].append(current_edge_index)
							current_edge_index += 1
					if vertex_1 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_1, edge_0]
						edge_dictionary[vertex_1] = [current_edge_index]
						current_edge_index += 1
					if vertex_2 in edge_dictionary:
						if len(edge_dictionary[vertex_2]) == 1:
							edge_List[current_edge_index] = [vertex_2, edge_1]
							edge_dictionary[vertex_2].append(current_edge_index)
					if vertex_2 not in edge_dictionary:
						edge_List[current_edge_index] = [vertex_2, edge_1]
						edge_dictionary[vertex_2] = [current_edge_index]
						current_edge_index += 1
					# creates entry for back two edges
					edge_List[current_edge_index] = [filtration_value, edge_2]
					edge_dictionary[vertex_3] = [current_edge_index]
					temp_edge_dictionary[vertex_3] = [current_edge_index]
					edge_List[current_edge_index+1] = [filtration_value, edge_3]
					edge_dictionary[vertex_3].append(current_edge_index + 1)
					temp_edge_dictionary[vertex_3].append(current_edge_index)
					current_edge_index += 2
					# creates entry for square
					# edge 0 and 1 are the last entry in the list of edge indices
					edge_0 = edge_dictionary[vertex_1][len(edge_dictionary[vertex_1])-1]
					edge_1 = edge_dictionary[vertex_2][len(edge_dictionary[vertex_2])-1]
					edge_2 = edge_dictionary[vertex_3][0]
					edge_3 = edge_dictionary[vertex_3][1]
					square_List[number_on_pixels] = [filtration_value, edge_0, edge_1, edge_2, edge_3]
					number_on_pixels += 1
					current_cube_index += 1
				else:
					temp_square_dictionary[current_cube_index] = -1
					square_dictionary[current_cube_index] = -1
					current_cube_index += 1
			# clears the dictionaries before moving on to next row to reduce memory
			vertex_dictionary = temp_vertex_dictionary
			edge_dictionary = temp_edge_dictionary
			square_dictionary = temp_square_dictionary
		# Removes the 0 tuples used to initiate edge and square lists
		vertex_List = vertex_List[:current_vertex_index]
		edge_List = edge_List[:current_edge_index]
		square_List = square_List[:number_on_pixels]
	# inner function to write perseus file
	def Write_Perseus_File(filtration, vertex_List, edge_List, square_List):
		# Constructs arrays to hold higher dimensional boundaries
		edge_Array = np.zeros((len(edge_List),6))
		square_Array = np.zeros((len(square_List),10))
		edge_iterator = 0
		square_iterator = 0
		for edge in edge_List:
			filtration_value = edge[0]
			# orients complex
			sign_vertex_0 = np.sign(edge[1][0])
			sign_vertex_1 = np.sign(edge[1][1])
			# removes sign used for orientation which is not stored as coefficient
			vertex0_index = abs(edge[1][0])
			vertex1_index = abs(edge[1][1])
			# edge entry is of the form [filtration value, # vertices, coefficient vertex1, vertex 1 index, coefficient vertex2, vertex 2 index]
			edge_Array[edge_iterator][:] = [filtration_value, 2, sign_vertex_0, vertex0_index, sign_vertex_1, vertex1_index]
			edge_iterator += 1
		for square in square_List:
			# This can probably be calculated explicitely in terms of the edge_List and number of cubes in complex
			filtration_value = square[0]
			edge0_index = square[1]
			edge1_index = square[2]
			edge2_index = square[3]
			edge3_index = square[4]
			# square entry is similar to form of edge entry
			square_Array[square_iterator][:] = [filtration_value, 4, 1, edge0_index, -1, edge1_index, -1, edge2_index, 1, edge3_index]
			square_iterator += 1
		# transforms perseus_Array into a text file
		os.chdir(perseus_dir)
		f = open(str(filename)+"_perseus_input_filtration_"+str(filtration)+".txt", "w+")
		# Perseus Array uses -1 to indicate change of dimension
		f.write(str(-1)+"\n")
		for vertex in vertex_List:
			f.write(str(int(vertex))+" "+str(0)+"\n")
		f.write(str(-1)+"\n")
		for edge in edge_Array:
			f.write(str(int(edge[0]))+" "+str(int(edge[1]))+" "+str(int(edge[2]))+" "+str(int(edge[3]))+" "+str(int(edge[4]))+" "+str(int(edge[5]))+"\n")
		f.write(str(-1)+"\n")
		for square in square_Array:
			f.write(str(int(square[0]))+" "+str(int(square[1]))+" "+str(int(square[2]))+" "+str(int(square[3]))+" "+str(int(square[4]))+" "+str(int(square[5]))+" "+str(int(square[6]))+" "+str(int(square[7]))+" "+str(int(square[8]))+" "+str(int(square[9]))+"\n")
		f.close()
		return()
	Write_Perseus_File(filtration, vertex_List, edge_List, square_List)
	return()


#switches back to the perseus_directory for computation of persistent homology
def Construct_Barcodes(filename, pixel_Matrix):
	os.chdir(perseus_dir)
	for filtration in range(2**(len(pixel_Matrix.shape))):    
		Construct_Lower_Dimensional_Faces(filename, filtration, pixel_Matrix)
		#constructs barcodes for filtration 'i'
		with open(os.devnull,"w") as devnull: 
			subprocess.check_call(["perseus", "cellcomp", str(filename)+"_perseus_input_filtration_"+str(filtration)+".txt", str(filename)+"_perseus_output_filtration_"+str(filtration)],stdout=devnull.fileno(),stderr=devnull.fileno())
	return


# Compute Change in Euler Characteristic for each pixel from barcode
def Euler_Characteristic_Change(filename, filtration, size_of_Image):
	os.chdir(perseus_dir)
	# creates an array to store [filtration value, betti 0, betti 1]
	euler_Lower_Star_Array = np.zeros((size_of_Image[0]*size_of_Image[1],2))
	f = open(str(filename)+"_perseus_output_filtration_"+str(filtration)+"_betti.txt", "r")
	read = f.readlines()
	# determines the length of a line to store betti numbers from file as array
	real_length_list = [int(i) for i in read[1].split()]
	betti_Number_Array = np.zeros((len(read[1:]),len(real_length_list)))
	# records information
	index = 0
	for betti_line in read[1:]:
		# betti numbers for filtration value
		betti_Number_Array[index][:] = [int(i) for i in betti_line.split()]
		index += 1
	f.close()
	# array used to compute change in Betti number to use to compute Euler Characteristic
	euler_Array_index = 0
	current_Betti_Numbers = np.zeros(len(size_of_Image))
	for line in betti_Number_Array:
		# stores position in first column of euler_Array (recall Perseus filtration values are position + 1)
		vertex_filtration_value = (line[0]-1)
		euler_Lower_Star_Array[euler_Array_index][0] = vertex_filtration_value
		euler_Char_Lower_Star = 0
		betti_index = 0
		#for betti_number in range(1,len(line)-1):
		for betti_number in line[1:-1]:
			euler_Char_Lower_Star += ((-1)**(betti_index))*(betti_number-current_Betti_Numbers[betti_index])
			current_Betti_Numbers[betti_index] = betti_number
			betti_index += 1
		euler_Lower_Star_Array[euler_Array_index][1] = int(euler_Char_Lower_Star)
		euler_Array_index += 1
	euler_Lower_Star_Array = euler_Lower_Star_Array[:index]
	return(euler_Lower_Star_Array)


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
		row = int(position  % (size_of_image[0] +1))
		moebius_Inversion_Array[row,column]+= (((-1)**filtration)*inversion_value)
	for information_pair in euler_Array1:
		# vertex filtration value
		filtration = 1
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		row = int(position // (size_of_image[1]+1))
		column = int(size_of_image[1]-1 - position % (size_of_image[1]+1))
		moebius_Inversion_Array[row,column+1] += (((-1)**filtration)*inversion_value)
	for information_pair in euler_Array2:
		# vertex filtration value
		filtration = 2
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		column = int(size_of_image[1]-1 - position // (size_of_image[0]+1))
		row = int(size_of_image[0]-1 - position % (size_of_image[0]+1))
		moebius_Inversion_Array[row+1,column+1] += (((-1)**filtration)*inversion_value)
	for information_pair in euler_Array3:
		# vertex filtration 
		filtration = 3
		position = int(information_pair[0])
		inversion_value = int(information_pair[1])
		row = int(size_of_image[0]-1 - position // (size_of_image[1]+1))
		column = int(position % (size_of_image[1]+1))
		moebius_Inversion_Array[row+1,column] += (((-1)**filtration)*inversion_value)
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
#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=2
import numpy as np


# Compute Euler Characteristic Curve from full cubes 
def Compute_Euler_Characteristic_Curves(voxel_Matrix):
    buffered_Image = np.zeros((voxel_Matrix.shape[0]+2,voxel_Matrix.shape[1]+2,voxel_Matrix.shape[2]+2))
    buffered_Image[1:-1,1:-1,1:-1] = voxel_Matrix[:,:,:]
    # indexes for number of critical vertices for respective ECC
    number_support_vertices_f0 = 0
    number_support_vertices_f1 = 0
    number_support_vertices_f2 = 0
    number_support_vertices_f3 = 0
    number_support_vertices_f4 = 0
    number_support_vertices_f5 = 0
    number_support_vertices_f6 = 0
    number_support_vertices_f7 = 0
    # stores [filtration value, nonzero Euler Characteristic]
    Euler_Characteristic_Curve_f0 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f1 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f2 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f3 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f4 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f5 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f6 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    Euler_Characteristic_Curve_f7 = np.zeros(((voxel_Matrix.shape[0]+1)*(voxel_Matrix.shape[1]+1)*(voxel_Matrix.shape[2]+1),2))
    # General Case (every cube will be on the interior of the grid after buffering image)
    column2 = 0
    for row_iterator in range(1,buffered_Image.shape[0]):
        for column1_iterator in range(1,buffered_Image.shape[1]):
            column2_iterator = 1
            cube0 = buffered_Image[row_iterator-1,column1_iterator-1,column2_iterator-1]
            cube1 = buffered_Image[row_iterator,column1_iterator-1,column2_iterator-1]
            cube2 = buffered_Image[row_iterator-1,column1_iterator,column2_iterator-1]
            cube3 = buffered_Image[row_iterator,column1_iterator,column2_iterator-1]
            for column2_iterator in range(1,buffered_Image.shape[2]):
                # cubes contained in the lower cross-section (filtrations are rotations counterclockwise with upwards pitch starting with a vector in the 1st octant)
                cube4 = buffered_Image[row_iterator-1,column1_iterator-1,column2_iterator]
                cube5 = buffered_Image[row_iterator,column1_iterator-1,column2_iterator]
                cube6 = buffered_Image[row_iterator-1,column1_iterator,column2_iterator]
                cube7 = buffered_Image[row_iterator,column1_iterator,column2_iterator]
                card_vertex = max(cube0, cube1, cube2, cube3, cube4, cube5, cube6, cube7)
                card_edge0_f0 = max(cube0, cube1, cube2, cube3)
                card_edge1_f0 = max(cube0, cube1, cube4, cube5)
                card_edge2_f0 = max(cube0, cube2, cube4, cube6)
                card_square0_f0 = max(cube0, cube1)
                card_square1_f0 = max(cube0, cube2)
                card_square2_f0 = max(cube0, cube4)
                card_cube_f0 = cube0
                card_edge0_f1 = card_edge0_f0
                card_edge1_f1 = max(cube1, cube3, cube5, cube7)
                card_edge2_f1 = card_edge1_f0
                card_square0_f1 = max(cube1, cube3)
                card_square1_f1 = card_square0_f0
                card_square2_f1 = max(cube1, cube5)
                card_cube_f1 = cube1
                card_edge0_f2 = card_edge0_f0
                card_edge1_f2 = max(cube2, cube3, cube6, cube7)
                card_edge2_f2 = card_edge1_f1
                card_square0_f2 = max(cube2, cube3)
                card_square1_f2 = card_square0_f1
                card_square2_f2 = max(cube3, cube7)
                card_cube_f2 = cube3
                card_edge0_f3 = card_edge0_f0
                card_edge1_f3 = card_edge2_f0
                card_edge2_f3 = card_edge1_f2
                card_square0_f3 = card_square1_f0
                card_square1_f3 = card_square0_f2
                card_square2_f3 = max(cube2, cube6)
                card_cube_f3 = cube2
                # cubes contained in the upper cross section (filtrations f4,f5,f6,f7 are reflections of filtrations f0,f1,f2,f3 respectively about the xy-plane)
                card_edge0_f4 = max(cube4, cube5, cube6, cube7)
                card_edge1_f4 = card_edge1_f0
                card_edge2_f4 = card_edge2_f0
                card_square0_f4 = max(cube4, cube5)
                card_square1_f4 = max(cube4, cube6)
                card_square2_f4 = card_square2_f0
                card_cube_f4 = cube4
                card_edge0_f5 = card_edge0_f4
                card_edge1_f5 = card_edge1_f1
                card_edge2_f5 = card_edge1_f0
                card_square0_f5 = max(cube5, cube7)
                card_square1_f5 = card_square0_f4
                card_square2_f5 = card_square2_f1
                card_cube_f5 = cube5
                card_edge0_f6 = card_edge0_f4
                card_edge1_f6 = card_edge1_f2
                card_edge2_f6 = card_edge1_f1
                card_square0_f6 = max(cube6, cube7)
                card_square1_f6 = card_square0_f5
                card_square2_f6 = card_square2_f2
                card_cube_f6 = cube7
                card_edge0_f7 = card_edge0_f4
                card_edge1_f7 = card_edge2_f0
                card_edge2_f7 = card_edge1_f2
                card_square0_f7 = card_square1_f4
                card_square1_f7 = card_square0_f6
                card_square2_f7 = card_square2_f3
                card_cube_f7 = cube6
                # computes euler_characteristic with respect to each filtration
                euler_characteristic_f0 = card_vertex - card_edge0_f0 - card_edge1_f0 - card_edge2_f0 + card_square0_f0 + card_square1_f0 + card_square2_f0 - card_cube_f0
                euler_characteristic_f1 = card_vertex - card_edge0_f1 - card_edge1_f1 - card_edge2_f1 + card_square0_f1 + card_square1_f1 + card_square2_f1 - card_cube_f1
                euler_characteristic_f2 = card_vertex - card_edge0_f2 - card_edge1_f2 - card_edge2_f2 + card_square0_f2 + card_square1_f2 + card_square2_f2 - card_cube_f2
                euler_characteristic_f3 = card_vertex - card_edge0_f3 - card_edge1_f3 - card_edge2_f3 + card_square0_f3 + card_square1_f3 + card_square2_f3 - card_cube_f3
                euler_characteristic_f4 = card_vertex - card_edge0_f4 - card_edge1_f4 - card_edge2_f4 + card_square0_f4 + card_square1_f4 + card_square2_f4 - card_cube_f4
                euler_characteristic_f5 = card_vertex - card_edge0_f5 - card_edge1_f5 - card_edge2_f5 + card_square0_f5 + card_square1_f5 + card_square2_f5 - card_cube_f5
                euler_characteristic_f6 = card_vertex - card_edge0_f6 - card_edge1_f6 - card_edge2_f6 + card_square0_f6 + card_square1_f6 + card_square2_f6 - card_cube_f6
                euler_characteristic_f7 = card_vertex - card_edge0_f7 - card_edge1_f7 - card_edge2_f7 + card_square0_f7 + card_square1_f7 + card_square2_f7 - card_cube_f7
                # cube and vertex filtration values
                if euler_characteristic_f0 != 0:
                    row = row_iterator
                    column1 = column1_iterator
                    column2 = column2_iterator
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*column2+buffered_Image.shape[0]*column1+row)+1
                    vertex_filtration_value = cube_value - column2*(buffered_Image.shape[0]*buffered_Image[1].shape[1])+ column1 + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*column2
                    Euler_Characteristic_Curve_f0[number_support_vertices_f0,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f0[number_support_vertices_f0,1] = euler_characteristic_f0
                    number_support_vertices_f0 += 1
                if euler_characteristic_f1 != 0:
                    row = row_iterator - 1
                    column1 = column1_iterator
                    column2 = column2_iterator
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*column2+(buffered_Image.shape[0]-row-1)*buffered_Image.shape[1]+column1)+1
                    vertex_filtration_value = cube_value - column2*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + (buffered_Image.shape[0]-1 - row) + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*column2
                    Euler_Characteristic_Curve_f1[number_support_vertices_f1,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f1[number_support_vertices_f1,1] = euler_characteristic_f1
                    number_support_vertices_f1 += 1
                if euler_characteristic_f2 != 0:
                    row = row_iterator - 1
                    column1 = column1_iterator - 1
                    column2 = column2_iterator
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*column2+(buffered_Image.shape[0]-row-1)+buffered_Image.shape[0]*(buffered_Image.shape[1]-column1-1))+1
                    vertex_filtration_value = cube_value - column2*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + (buffered_Image.shape[1]-1 - column1) + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*column2
                    Euler_Characteristic_Curve_f2[number_support_vertices_f2,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f2[number_support_vertices_f2,1] = euler_characteristic_f2
                    number_support_vertices_f2 += 1
                if euler_characteristic_f3 != 0:
                    row = row_iterator
                    column1 = column1_iterator - 1
                    column2 = column2_iterator
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*column2+buffered_Image.shape[1]*row+(buffered_Image.shape[1]-column1-1))+1
                    vertex_filtration_value = cube_value - column2*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + row + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*column2
                    Euler_Characteristic_Curve_f3[number_support_vertices_f3,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f3[number_support_vertices_f3,1] = euler_characteristic_f3
                    number_support_vertices_f3 += 1
                if euler_characteristic_f4 != 0:
                    row = row_iterator
                    column1 = column1_iterator
                    column2 = column2_iterator - 1
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*(buffered_Image.shape[2]-1-column2)+buffered_Image.shape[0]*column1+row)+1
                    vertex_filtration_value = cube_value - (buffered_Image.shape[2]-1-column2)*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + column1 + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*(buffered_Image.shape[2]-1-column2)
                    Euler_Characteristic_Curve_f4[number_support_vertices_f4,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f4[number_support_vertices_f4,1] = euler_characteristic_f4
                    number_support_vertices_f4 += 1
                if euler_characteristic_f5 != 0:
                    row = row_iterator - 1
                    column1 = column1_iterator
                    column2 = column2_iterator - 1
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*(buffered_Image.shape[2]-1-column2)+(buffered_Image.shape[0]-row-1)*buffered_Image.shape[1]+column1)+1
                    vertex_filtration_value = cube_value - (buffered_Image.shape[2]-1-column2)*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + (buffered_Image.shape[0]-1 - row) + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*(buffered_Image.shape[2]-1-column2)
                    Euler_Characteristic_Curve_f5[number_support_vertices_f5,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f5[number_support_vertices_f5,1] = euler_characteristic_f5
                    number_support_vertices_f5 += 1
                if euler_characteristic_f6 != 0:
                    row = row_iterator - 1
                    column1 = column1_iterator - 1
                    column2 = column2_iterator - 1
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*(buffered_Image.shape[2]-1-column2)+(buffered_Image.shape[0]-row-1)+buffered_Image.shape[0]*(buffered_Image.shape[1]-column1-1))+1
                    vertex_filtration_value = cube_value - (buffered_Image.shape[2]-1-column2)*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + (buffered_Image.shape[1]-1 - column1) + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*(buffered_Image.shape[2]-1-column2)
                    Euler_Characteristic_Curve_f6[number_support_vertices_f6,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f6[number_support_vertices_f6,1] = euler_characteristic_f6
                    number_support_vertices_f6 += 1
                if euler_characteristic_f7 != 0:
                    row = row_iterator
                    column1 = column1_iterator - 1
                    column2 = column2_iterator - 1
                    cube_value = (buffered_Image.shape[0]*buffered_Image.shape[1]*(buffered_Image.shape[2]-1-column2)+buffered_Image.shape[1]*row+(buffered_Image.shape[1]-column1-1))+1
                    vertex_filtration_value = cube_value - (buffered_Image.shape[2]-1-column2)*(buffered_Image.shape[0]*buffered_Image[1].shape[1]) + row + (buffered_Image.shape[0]+1)*(buffered_Image.shape[1]+1)*(buffered_Image.shape[2]-1-column2)
                    Euler_Characteristic_Curve_f7[number_support_vertices_f7,0] = vertex_filtration_value
                    Euler_Characteristic_Curve_f7[number_support_vertices_f7,1] = euler_characteristic_f7
                    number_support_vertices_f7 += 1
                # updates sliding window with known entries
                cube0 = cube4
                cube1 = cube5
                cube2 = cube6
                cube3 = cube7
    # tirms to support of the Euler_Characteristic_Curves
    Euler_Characteristic_Curve_f0 = Euler_Characteristic_Curve_f0[:number_support_vertices_f0]
    Euler_Characteristic_Curve_f1 = Euler_Characteristic_Curve_f1[:number_support_vertices_f1]
    Euler_Characteristic_Curve_f2 = Euler_Characteristic_Curve_f2[:number_support_vertices_f2]
    Euler_Characteristic_Curve_f3 = Euler_Characteristic_Curve_f3[:number_support_vertices_f3]
    Euler_Characteristic_Curve_f4 = Euler_Characteristic_Curve_f4[:number_support_vertices_f4]
    Euler_Characteristic_Curve_f5 = Euler_Characteristic_Curve_f5[:number_support_vertices_f5]
    Euler_Characteristic_Curve_f6 = Euler_Characteristic_Curve_f6[:number_support_vertices_f6]
    Euler_Characteristic_Curve_f7 = Euler_Characteristic_Curve_f7[:number_support_vertices_f7]
    return(Euler_Characteristic_Curve_f0, Euler_Characteristic_Curve_f1, Euler_Characteristic_Curve_f2, Euler_Characteristic_Curve_f3, Euler_Characteristic_Curve_f4, Euler_Characteristic_Curve_f5, Euler_Characteristic_Curve_f6, Euler_Characteristic_Curve_f7)


# Constructs the moebius_Inversion_Matrix from Euler Characteristic Curves
def generate_Inversion_Values(voxel_Matrix, euler_Array0, euler_Array1, euler_Array2, euler_Array3, euler_Array4, euler_Array5, euler_Array6, euler_Array7):
    size_of_image = (voxel_Matrix.shape[0]+2, voxel_Matrix.shape[1]+2, voxel_Matrix.shape[2]+2)
    # Allows for placement into extra row and column to account for possible non-zero values which is later trimmed
    moebius_Inversion_Array = np.zeros((size_of_image[0], size_of_image[1], size_of_image[2]), dtype='int')
    # Takes alternating sum of lower star Euler Characteristic for appropriate pixel
    for information_pair in euler_Array0:
        # vertex filtration value
        filtration = 0
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[1]+1))
        vertex_column1 = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) // (size_of_image[1]+1))
        vertex_column2 = int(position // ((size_of_image[0]+1)*(size_of_image[1]+1)))
        row = vertex_row
        column1 = vertex_column1
        column2 = vertex_column2
        moebius_Inversion_Array[row,column1,column2]+= (((-1)**filtration)*inversion_value)
    for information_pair in euler_Array1:
        # vertex filtration value
        filtration = 1
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int(size_of_image[0] - (position % ((size_of_image[0]+1)*(size_of_image[1]+1)) // (size_of_image[1]+1)))
        vertex_column1 = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[1]+1))
        vertex_column2 = int(position // ((size_of_image[0]+1)*(size_of_image[1]+1)))
        row = vertex_row - 1
        column1 = vertex_column1
        column2 = vertex_column2
        moebius_Inversion_Array[row+1,column1,column2] += (((-1)**filtration)*inversion_value)
    for information_pair in euler_Array2:
        # vertex filtration value
        filtration = 2
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int(size_of_image[0] - ((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[0]+1)))
        vertex_column1 = int(size_of_image[1] - ((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) // (size_of_image[0]+1)))
        vertex_column2 = int(position // ((size_of_image[0]+1)*(size_of_image[1]+1)))
        row = vertex_row - 1
        column1 = vertex_column1 - 1
        column2 = vertex_column2
        moebius_Inversion_Array[row+1,column1+1,column2] += (((-1)**filtration)*inversion_value)
    for information_pair in euler_Array3:
        # vertex filtration 
        filtration = 3
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) // (size_of_image[1]+1))
        vertex_column1 = int(size_of_image[1] - ((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[1]+1)))
        vertex_column2 = int(position // ((size_of_image[0]+1)*(size_of_image[1]+1)))
        row = vertex_row
        column1 = vertex_column1 - 1
        column2 = vertex_column2
        moebius_Inversion_Array[row,column1+1,column2] += (((-1)**filtration)*inversion_value)
    for information_pair in euler_Array4:
        # vertex filtration value
        filtration = 4
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[1]+1))
        vertex_column1 = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) // (size_of_image[1]+1))
        vertex_column2 = int(size_of_image[2] - (position // ((size_of_image[0]+1)*(size_of_image[1]+1))))
        row = vertex_row
        column1 = vertex_column1
        column2 = vertex_column2 - 1
        moebius_Inversion_Array[row,column1,column2+1]+= ((-1)**(filtration+1)*inversion_value)
    for information_pair in euler_Array5:
        # vertex filtration value
        filtration = 5
        position = int(information_pair[0])
        inversion_value = int(information_pair[1])
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int(size_of_image[0] - (position % ((size_of_image[0]+1)*(size_of_image[1]+1)) // (size_of_image[1]+1)))
        vertex_column1 = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[1]+1))
        vertex_column2 = int(size_of_image[2] - (position // ((size_of_image[0]+1)*(size_of_image[1]+1))))
        row = vertex_row - 1
        column1 = vertex_column1
        column2 = vertex_column2
        moebius_Inversion_Array[row+1,column1,column2] += ((-1)**(filtration+1)*inversion_value)
    for information_pair in euler_Array6:
        # vertex filtration value
        filtration = 6
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int(size_of_image[0] - ((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[0]+1)))
        vertex_column1 = int(size_of_image[1] - ((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) // (size_of_image[0]+1)))
        vertex_column2 = int(size_of_image[2] - (position // ((size_of_image[0]+1)*(size_of_image[1]+1))))
        row = vertex_row - 1
        column1 = vertex_column1 - 1
        column2 = vertex_column2
        moebius_Inversion_Array[row+1,column1+1,column2] += ((-1)**(filtration+1)*inversion_value)
    for information_pair in euler_Array7:
        # vertex filtration 
        filtration = 7
        position = int(information_pair[0]-1)
        inversion_value = int(information_pair[1])
        vertex_row = int((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) // (size_of_image[1]+1))
        vertex_column1 = int(size_of_image[1] - ((position % ((size_of_image[0]+1)*(size_of_image[1]+1))) % (size_of_image[1]+1)))
        vertex_column2 = int(size_of_image[2] - (position // ((size_of_image[0]+1)*(size_of_image[1]+1))))
        row = vertex_row
        column1 = vertex_column1 - 1
        column2 = vertex_column2
        moebius_Inversion_Array[row,column1+1,column2] += ((-1)**(filtration+1)*inversion_value)
    moebius_Inverted_Matrix = moebius_Inversion_Array[1:-1,1:-1,1:-1]
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
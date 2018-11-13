#Topological Voxel Reconstruction
#Reconstructs Binary images for dimension d=3
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


# 3D visualization of voxel data
from mpl_toolkits.mplot3d import Axes3D
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


# 3D visualization of voxel data pipeline
# normalizes grayscale voxel values
def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


# rescales image for faster rendering
def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean


# plots voxels on 3-dimensional grid
def plot_cube(cube, img_Dim, angle=320):
    # helper function which seperates certain voxels to improve visualization
    def explode(data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3]*2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    # helper function to move specific voxels to improve visualization
    def expand_coordinates(indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z
    cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=img_Dim*2)
    ax.set_ylim(top=img_Dim*2)
    ax.set_zlim(top=img_Dim*2)
    
    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.show()
    return
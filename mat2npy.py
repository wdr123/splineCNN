import numpy as np
from scipy import io as sio


def mat2npy_both(file_path):
    
    mat = sio.loadmat(file_path) # or: np.loadmat(...) data为'dict'类型数据（字典型）

    tetras = mat['face0'] - 1 # matlab index from 1 but python from 0, so transform from matlab index to python index
    points= mat['point0']

    set_edge = set([])

    for tetra in tetras:
        set_edge.add((tetra[0],tetra[1]))
        set_edge.add((tetra[0],tetra[2]))
        set_edge.add((tetra[0],tetra[3]))
        set_edge.add((tetra[1],tetra[2]))
        set_edge.add((tetra[1],tetra[3]))
        set_edge.add((tetra[2],tetra[3]))

    set_edge = np.array(list(set_edge))
    left_index = []
    right_index = []
    for left, right in zip(set_edge[:,0],set_edge[:,1]):
        left_index.append(left)
        right_index.append(right)

    edge_idnex = np.array([left_index, right_index])

    return points, edge_idnex


def mat2npy_point(file_path):
    
    mat = sio.loadmat(file_path)

    points= mat['point0']

    return points

def mat2npy_sparse(posterior_path, srf_path, vessel_contour_path):
    
    mat1 = sio.loadmat(posterior_path) 
    points1= mat1['point0']
    mat2 = sio.loadmat(srf_path) 
    points2= mat2['point0']
    mat3 = sio.loadmat(vessel_contour_path) 
    points3= mat3['point0']

    points = np.concatenate([points1, points2, points3], axis=0)

    return points








# mat = sio.loadmat('raw_pc/1-L/1-L_01242014-USPlane0/posterior_points.mat')
# print(mat['point0'].shape)
# np.save('./file.npy',data) # 保存npy文件
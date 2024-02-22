import os
import numpy as np
from scipy import io as sio
import random



# data_path = 'soheil_data/1-L/1-L_01242014-USPlane0+1+9'
# file_path = os.path.join(data_path, 'GT.mat')
# data_path1 = 'soheil_data/1-L/1-L_01242014-USPlane0+1+9/IntraOperative'
# file_path1 = os.path.join(data_path1, 'posterior_points_down.mat')


# mat = sio.loadmat(file_path1) # or: np.loadmat(...) data为'dict'类型数据（字典型）

# # tetras = mat['face0']
# points= mat['point0']

# # print(tetras.shape)
# print(points.shape)

# edge_path = os.path.join(data_path, 'edges.npy')

# points = np.load(point_path)
# edges = np.array(np.load(edge_path), dtype='int64')
# edges = edges - 1 # matlab index from 1 but python from 0, so transform from matlab index to python index

random.seed(1)
a = [1,2,3,4,5,6]
random.shuffle(a) 
print(a)



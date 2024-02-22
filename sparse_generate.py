import os
import numpy as np
from mat2npy import mat2npy_point, mat2npy_sparse

source_root = 'soheil_data'
result_root = 'sparse_coarsen'

# for dirname in os.listdir(source_root):
#     if dirname[1] == '-':
#         result_dir = os.path.join(result_root, dirname)
#         source_dir = os.path.join(source_root, dirname)

#         for dirname1 in os.listdir(source_dir):
            
#             if dirname1[3] == '_':
#                 data_dir = os.path.join(result_dir, dirname1) # Directory to store each data instance
#                 intra_dir = os.path.join(source_dir, dirname1) # Directory for extracting intra data and ground-truth data

#                 posterior_path = os.path.join(intra_dir, 'IntraOperative','posterior_points_down.mat')
#                 srf_path = os.path.join(intra_dir, 'IntraOperative','srf_points_down.mat')
#                 vessel_contour_path = os.path.join(intra_dir, 'IntraOperative','vessel_contour_points_down.mat')
#                 gt_path = os.path.join(intra_dir, 'GT.mat') vessel_deformed

#                 sparse_points = mat2npy_sparse(posterior_path, srf_path, vessel_contour_path)
#                 gt_points = mat2npy_point(gt_path)

#                 if not os.path.exists(data_dir):
#                     os.makedirs(data_dir)

#                 np.save(os.path.join(data_dir, 'sparse_points.npy'), sparse_points) # 保存npy文件
#                 np.save(os.path.join(data_dir, 'gt_points.npy'), gt_points) # 保存npy文件


for dirname in os.listdir(os.path.join(result_root,'1-L')):
    gt_path = os.path.join(os.path.join(result_root,'1-L',dirname,'gt_points.npy'))
    gt = np.load(gt_path)
    sparse_path = os.path.join(os.path.join(result_root,'1-L',dirname,'sparse_points.npy'))
    sparse = np.load(sparse_path)

    print(gt.shape, sparse.shape)





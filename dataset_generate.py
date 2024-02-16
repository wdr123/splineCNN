import os
import numpy as np
from mat2npy import mat2npy_both, mat2npy_point

source_root = 'raw_coarsen'
result_root = 'data_coarsen'

for dirname in os.listdir(source_root):
    if dirname[1] == '-':
        result_dir = os.path.join(result_root, dirname)
        source_dir = os.path.join(source_root, dirname)

        pre_dir = os.path.join(source_dir, 'PreOperative')
        prefile_path = os.path.join(pre_dir, 'rc_cor_tet.mat')
        pre_points, pre_edges = mat2npy_both(prefile_path)

        for dirname1 in os.listdir(source_dir):
            
            if dirname1[3] == '_':
                data_dir = os.path.join(result_dir, dirname1) # Directory to store each data instance
                intra_dir = os.path.join(source_dir, dirname1) # Directory for extracting intra data and ground-truth data

                libr_path = os.path.join(intra_dir, 'IntraOperative','rc_cor_tet.mat')
                gt_path = os.path.join(intra_dir, 'rc_cor_tet.mat')

                libr_points = mat2npy_point(libr_path)
                gt_points = mat2npy_point(gt_path)

                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                np.save(os.path.join(data_dir, 'points.npy'), [pre_points, libr_points, gt_points]) # 保存npy文件
                np.save(os.path.join(data_dir, 'edges.npy'), pre_edges) # 保存npy文件





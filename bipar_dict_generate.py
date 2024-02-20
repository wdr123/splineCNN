import numpy as np
import json
import os

sparse_root = 'sparse_coarsen/1-L'
gt_root = 'data_coarsen/1-L'
top = 2

def nearest_neighbor(sparse_point, gt_points, top=2):
    distance = np.sqrt(np.sum(np.square(gt_points - sparse_point), axis=1))
    indices = np.argsort(distance)
    gt_ids = indices[:top]

    return gt_ids



for dirname in os.listdir(os.path.join(sparse_root)):
    nearest = {}
    gt_path = os.path.join(os.path.join(gt_root,dirname,'points.npy'))
    points = np.load(gt_path)
    gt = points[2]
    sparse_path = os.path.join(os.path.join(sparse_root,dirname,'sparse_points.npy'))
    sparse = np.load(sparse_path)
    # print(gt.shape, sparse.shape)

    for idx, sparse_point in enumerate(sparse):
        gt_ids = nearest_neighbor(sparse_point, gt, top=top) 
        
        for gt_id in gt_ids:
            gt_id = int(gt_id)
            if gt_id not in nearest.keys():
                nearest[gt_id] = [idx]
            else:
                nearest[gt_id].append(idx)

    with open(os.path.join(sparse_root,dirname,'nearest.json'), "w+") as file:
        json.dump(nearest, file)

        




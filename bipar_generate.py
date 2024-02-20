import numpy as np
import json
import os

sparse_root = 'sparse_coarsen/1-L'
gt_root = 'data_coarsen/1-L'


for dirname in os.listdir(os.path.join(sparse_root)):
    nearest_path = os.path.join(sparse_root,dirname,'nearest.json')
    f = open(nearest_path, "r")
    nearest = json.load(f)
    f.close()

    gt_path = os.path.join(os.path.join(gt_root,dirname,'points.npy'))
    points = np.load(gt_path)
    gt = points[2]
    sparse_path = os.path.join(os.path.join(sparse_root,dirname,'sparse_points.npy'))
    sparse = np.load(sparse_path)
    bipar_points = sparse.copy()
    sparse_gt_map = {}

    # print(gt.shape, sparse.shape)

    for idx, gt_id in enumerate(nearest):
        gt_id_int = int(gt_id)
        gt_sparse_id = len(bipar_points)
        bipar_points = np.append(bipar_points, [gt[gt_id_int]], axis=0)
        sparse_ids = np.array(nearest[gt_id])[:, np.newaxis]
        gt_ids = np.ones_like(sparse_ids)
        gt_ids.fill(gt_sparse_id)
        new_edges = np.concatenate([sparse_ids, gt_ids], axis=-1)
        if idx == 0:
            bipar_edges = new_edges
        else:
            bipar_edges = np.concatenate([bipar_edges, new_edges], axis=0)

        sparse_gt_map[gt_id] = gt_sparse_id
        # print(bipar_points.shape, bipar_edges.shape)

    sparse_gt_map_path = os.path.join(sparse_root,dirname,'sparse_gt_map.json')
    bipar_point_path = os.path.join(sparse_root,dirname,'points.npy')
    bipar_edge_path = os.path.join(sparse_root,dirname,'edges.npy')

    np.save(bipar_point_path, bipar_points)
    np.save(bipar_edge_path, bipar_edges)

    with open(sparse_gt_map_path, "w+") as file:
        json.dump(sparse_gt_map, file)
        
        
        




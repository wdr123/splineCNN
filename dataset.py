import os
import numpy as np
import torch
import random
import json
import itertools

'''gt_id == dense_id in dense graph, bipar_gt_id == dense_id in bipartite graph, bipar_sparse_id == sparse_id in bipartite graph'''
'''bipartite graph: point=={sparse_id+bipar_gt_id}, edge=={sparse_id2bipar_gt_id}'''


class VUDense(object):
    def __init__(self, root_dir = "data_coarsen", seed = 0, train=True, bs=6):

        self.train_list = []
        self.test_list = []
        self.train = train
        self.sparse_root_dir = "sparse_coarsen"
        self.dense_points = {}
        self.dense_edges = {}
        self.bipar_points = {}
        self.bipar_embeddings = {}
        # self.bipar_edges = {}
        self.map_gt2bipargt = {}
        self.map_gt2sparse = {}
        self.batch_size = bs
        self.index = 0
        self.liver = None
        self.liver_mini_edgeP_num = {}

        random.seed(seed)

        for liver_name in os.listdir(root_dir):
            liver_dir = os.path.join(root_dir, liver_name)
            bipar_dir = os.path.join(self.sparse_root_dir, liver_name)
            register_names = os.listdir(liver_dir)

            random.shuffle(register_names) 
       
            if self.train:
                train_register_names = register_names[:int(0.75*len(register_names))] # Train:Test=3:1
                for register_name in train_register_names:
                    self.train_list.append(register_name)
                    dense_register_path = os.path.join(liver_dir, register_name)
                    bipar_register_path = os.path.join(bipar_dir, register_name)

                    dense_point_path = os.path.join(dense_register_path, 'points.npy')
                    dense_edge_path = os.path.join(dense_register_path, 'edges.npy')
                    bipar_point_path = os.path.join(bipar_register_path, 'points.npy')
                    bipar_embed_path = os.path.join(bipar_register_path, 'embedding.npy')
                    # bipar_edge_path = os.path.join(bipar_register_path, 'edges.npy')
                    gt_id2bipar_gt_id_path = os.path.join(bipar_register_path, 'sparse_gt_map.json')
                    gt_id2bipar_sparse_id_path = os.path.join(bipar_register_path, 'nearest.json')
                    

                    self.dense_points[register_name] = np.load(dense_point_path)
                    self.dense_edges[register_name] = np.load(dense_edge_path)
                    self.dense_edges[register_name] = np.swapaxes(self.dense_edges[register_name],0,1)
                    self.bipar_points[register_name] = np.load(bipar_point_path)
                    self.bipar_embeddings[register_name] = torch.load(bipar_embed_path)
                    
                    # self.bipar_edges[register_name] = np.load(bipar_edge_path)
                    with open(gt_id2bipar_gt_id_path, "r") as f:
                        self.map_gt2bipargt[register_name] = json.load(f)

                    if liver_name not in self.liver_mini_edgeP_num:
                        self.liver_mini_edgeP_num[liver_name] = len(self.map_gt2bipargt[register_name])
                    else:
                        if self.liver_mini_edgeP_num[liver_name] > len(self.map_gt2bipargt[register_name]):
                            self.liver_mini_edgeP_num[liver_name] = len(self.map_gt2bipargt[register_name])

                    with open(gt_id2bipar_sparse_id_path, "r") as f:
                        self.map_gt2sparse[register_name] = json.load(f)
            else:    
                test_register_names = register_names[int(0.75*len(register_names)):] # Train:Test=3:1
                for register_name in test_register_names:
                    self.test_list.append(register_name)
                    dense_register_path = os.path.join(liver_dir, register_name)
                    bipar_register_path = os.path.join(bipar_dir, register_name)

                    dense_point_path = os.path.join(dense_register_path, 'points.npy')
                    dense_edge_path = os.path.join(dense_register_path, 'edges.npy')
                    bipar_point_path = os.path.join(bipar_register_path, 'points.npy')
                    bipar_embed_path = os.path.join(bipar_register_path, 'embedding.npy')
                    # bipar_edge_path = os.path.join(bipar_register_path, 'edges.npy')
                    gt_id2bipar_gt_id_path = os.path.join(bipar_register_path, 'sparse_gt_map.json')
                    gt_id2bipar_sparse_id_path = os.path.join(bipar_register_path, 'nearest.json')

                    self.dense_points[register_name] = np.load(dense_point_path)
                    self.dense_edges[register_name] = np.load(dense_edge_path)
                    self.dense_edges[register_name] = np.swapaxes(self.dense_edges[register_name],0,1)
                    # print(self.dense_edges[register_name].shape)
                    self.bipar_points[register_name] = np.load(bipar_point_path)
                    self.bipar_embeddings[register_name] = torch.load(bipar_embed_path)
                    # self.bipar_edges[register_name] = np.load(bipar_edge_path)
                    with open(gt_id2bipar_gt_id_path, "r") as f:
                        self.map_gt2bipargt[register_name] = json.load(f)
                    
                    if liver_name not in self.liver_mini_edgeP_num:
                        self.liver_mini_edgeP_num[liver_name] = len(self.map_gt2bipargt[register_name])
                    else:
                        if self.liver_mini_edgeP_num[liver_name] > len(self.map_gt2bipargt[register_name]):
                            self.liver_mini_edgeP_num[liver_name] = len(self.map_gt2bipargt[register_name])

                    with open(gt_id2bipar_sparse_id_path, "r") as f:
                        self.map_gt2sparse[register_name] = json.load(f)

        

    def __len__(self):
        
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)
        

    def __getitem__(self, idx):

        flag = False

        if self.index == self.batch_size:
            self.index = 0

        while(not flag):
            if self.train:
                register_name = self.train_list[idx]
            else:
                register_name = self.test_list[idx]

            if self.index == 0:
                self.liver = register_name[:3]    
                flag = True
            else:
                if register_name[:3] == self.liver:
                    flag = True
                else:
                    flag = False
                    idx = random.randrange(self.__len__())

        self.index += 1

        dense_points = self.dense_points[register_name]
        dense_edges = self.dense_edges[register_name].astype(int, copy=False)
        bipar_points = self.bipar_points[register_name]
        bipar_embeddings = self.bipar_embeddings[register_name]
        # bipar_edges = self.bipar_edges[register_name]
        gt2bipar_gt = self.map_gt2bipargt[register_name]
        truncate_number_batch = self.liver_mini_edgeP_num[self.liver]
        # print(self.liver)
        # print(truncate_number_batch)
        gt2bipar_gt = dict(itertools.islice(gt2bipar_gt.items(), truncate_number_batch)) 
        # print(len(gt2bipar_gt))

        gt2bipar_sparse = self.map_gt2sparse[register_name]

        pre_points = dense_points[0]
        libr_points = dense_points[1]
        gt_points = torch.tensor(dense_points[2], dtype=torch.float, requires_grad=False)

        x = np.concatenate([pre_points, libr_points], axis=1) # 2044 nodes with 6 (3+3) features each
        dense_input = torch.tensor(x, dtype=torch.float)


        dense_edges = torch.tensor(dense_edges, requires_grad=False)  # 18898 edges
        # print(dense_edges.shape)

        assert dense_input.size(0) == gt_points.size(0)

        sparse_embeddings = []
        edge_points = []
        sparse_supervision = []

        for gt_id in gt2bipar_gt: # gt_id type: str
            # print(bipar_embeddings.shape)
            sparse_embeddings.append(bipar_embeddings[gt2bipar_gt[gt_id]]) # gt2bipar_gt is nearest map which maps dense id to dense id in bipartite graph.
            # bipar_embeddings is a tensor of size (num_point, 1024)
            edge_points.append(int(gt_id))
            sparse_supervision.append(torch.from_numpy(np.average(bipar_points[gt2bipar_sparse[gt_id]],axis=0)))

        # print(len(sparse_embeddings))
        sparse_embeddings = torch.stack(sparse_embeddings).float().clone().detach().requires_grad_(False)
        edge_points = torch.tensor(edge_points, dtype=torch.int, requires_grad=False)
        sparse_supervision = torch.stack(sparse_supervision).float().clone().detach().requires_grad_(False)
        
        sample = {'dense_points': dense_input, 'dense_edges': dense_edges, 'dense_gt': gt_points,\
                   'sparse_embedding': sparse_embeddings, 'edge_points': edge_points, 'sparse_supervision': sparse_supervision}

        return sample
    



class VUSparse(object):
    def __init__(self, root_dir = "sparse_coarsen"):

        self.bipar_points = {}
        self.bipar_edges = {}
        self.train_list = []


        for liver_name in os.listdir(root_dir):
            liver_dir = os.path.join(root_dir, liver_name)
            register_names = os.listdir(liver_dir)
       
            for register_name in register_names:
                self.train_list.append(register_name)
                sparse_register_path = os.path.join(liver_dir, register_name)

                bipar_point_path = os.path.join(sparse_register_path, 'points.npy')
                bipar_edge_path = os.path.join(sparse_register_path, 'edges.npy')

                self.bipar_points[register_name] = np.load(bipar_point_path)
                self.bipar_edges[register_name] = np.load(bipar_edge_path)
        

    def __len__(self):
        
        return len(self.train_list)

        

    def __getitem__(self, idx):

        register_name = self.train_list[idx]
        bipar_points = self.bipar_points[register_name]
        bipar_edges = self.bipar_edges[register_name]

        bipar_points = torch.tensor(bipar_points, requires_grad=False)
        bipar_edges = torch.tensor(bipar_edges, requires_grad=False)
        
        sample = {'bipar_points': bipar_points, 'bipar_edges': bipar_edges, 'register_name': register_name}

        return sample
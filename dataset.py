import os
import numpy as np
import torch
import random
import json


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
        self.map_gt2bipar = {}
        self.map_gt2sparse = {}
        self.batch_size = bs
        self.index = 0
        self.liver = None

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
                    self.bipar_points[register_name] = np.load(bipar_point_path)
                    self.bipar_embeddings[register_name] = np.load(bipar_embed_path)
                    # self.bipar_edges[register_name] = np.load(bipar_edge_path)
                    with open(gt_id2bipar_gt_id_path, "r") as f:
                        self.map_gt2bipar[register_name] = json.load(f)
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
                    self.bipar_points[register_name] = np.load(bipar_point_path)
                    self.bipar_embeddings[register_name] = np.load(bipar_embed_path)
                    # self.bipar_edges[register_name] = np.load(bipar_edge_path)
                    with open(gt_id2bipar_gt_id_path, "r") as f:
                        self.map_gt2bipar[register_name] = json.load(f)
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

        while(~flag):
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
        dense_edges = self.dense_edges[register_name]
        bipar_points = self.bipar_points[register_name]
        bipar_embeddings = self.bipar_embeddings[register_name]
        # bipar_edges = self.bipar_edges[register_name]
        gt2bipar_gt = self.map_gt2bipar[register_name]
        gt2bipar_sparse = self.map_gt2sparse[register_name]

        pre_points = dense_points[0]
        libr_points = dense_points[1]
        gt_points = torch.tensor(dense_points[2], dtype=torch.float, requires_grad=False)

        x = np.concatenate([pre_points, libr_points], axis=1) # 2044 nodes with 6 (3+3) features each
        dense_input = torch.tensor(x, dtype=torch.float)

        dense_edges = torch.tensor(dense_edges, requires_grad=False)  # 18898 edges

        assert dense_input.size(0) == gt_points.size(0)

        sparse_embeddings = []
        edge_points = []
        sparse_supervision = []

        for gt_id in gt2bipar_gt:
            sparse_embeddings.append(bipar_embeddings[gt2bipar_gt[gt_id]])
            edge_points.append(gt_id)
            sparse_supervision.append(np.average(bipar_points[gt2bipar_sparse[gt_id]],axis=0))

        sparse_embeddings = torch.tensor(sparse_embeddings, dtype=torch.float, requires_grad=False)
        edge_points = torch.tensor(edge_points, dtype=torch.float, requires_grad=False)
        sparse_supervision = torch.tensor(sparse_supervision, dtype=torch.float, requires_grad=False)
       
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
        
        sample = {'bipar_points': bipar_points, 'bipar_edges': bipar_edges}

        return sample
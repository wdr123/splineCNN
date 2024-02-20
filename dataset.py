import os
import pandas as pd
import numpy as np
import torch
import random


class VUDataset(object):
    def __init__(self, root_dir = "data_coarsen", seed = 0, train = True):

        self.train_dirs = {}
        self.train_list = []
        self.test_dirs = {}
        self.test_list = []
        self.train = train

        random.seed(seed)

        for liver_name in os.listdir(root_dir):
            liver_dir = os.path.join(root_dir, liver_name)
            register_names = os.listdir(liver_dir)

            random.shuffle(register_names) 

            if train:
                register_names = register_names[:int(0.75*len(register_names))] # Train:Test=3:1
                self.train_list.extend([os.path.join(liver_dir, register_name) for register_name in register_names])
                if liver_name not in self.train_dirs:
                    self.train_dirs[liver_name] = register_names
            else:
                register_names = register_names[int(0.75*len(register_names)):] # Train:Test=3:1
                self.test_list.extend([os.path.join(liver_dir, register_name) for register_name in register_names])
                if liver_name not in self.test_dirs:
                    self.test_dirs[liver_name] = register_names

        

    def __len__(self):
        
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)
        

    def __getitem__(self, idx):
        if self.train:
            read_folder_path = self.train_list[idx]
        else:
            read_folder_path = self.test_list[idx]

        data_path = read_folder_path
        point_path = os.path.join(data_path, 'points.npy')
        edge_path = os.path.join(data_path, 'edges.npy')

        points = np.load(point_path)
        edges = np.array(np.load(edge_path), dtype='int64')
        edges = edges - 1 # matlab index from 1 but python from 0, so transform from matlab index to python index

        pre_points = points[0]
        libr_points = points[1]
        gt_points = torch.tensor(points[2], dtype=torch.float, requires_grad=False)

        x = np.concatenate([pre_points, libr_points], axis=1) # 2044 nodes with 6 (3+3) features each
        x = torch.tensor(x, dtype=torch.float)

        edge_index = torch.tensor(edges)  # 18898 edges

        assert x.size(0) == gt_points.size(0)
        
        sample = {'points': x, 'edges': edge_index, 'gt': gt_points}

        return sample

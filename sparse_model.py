import torch
import math
import os
import numpy as np
from torch_spline_conv import spline_conv


class SplineConv(torch.nn.Module):
    def __init__(self, input_channel=3, embed_channel=1024, output_channel=3, num_kernel=5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.embed_channel = embed_channel
        self.num_kernel = num_kernel

        self.root_weight = torch.nn.Parameter(torch.rand((input_channel, embed_channel), dtype=torch.double))
        self.weight = torch.nn.Parameter(torch.rand((25, input_channel, embed_channel), dtype=torch.double))
        self.weight1 = torch.nn.Parameter(torch.rand((25, embed_channel, output_channel), dtype=torch.double))  # 30 parameters for in_channels x out_channels
        self.root_weight1 = torch.nn.Parameter(torch.rand((embed_channel, output_channel), dtype=torch.double))  # separately weight root nodes

        self.kernel_size = torch.tensor([num_kernel, num_kernel])  # 5 parameters in each edge dimension
        self.is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)  # only use open B-splines
        self.degree = 1  # B-spline degree of 1
        self.norm = True  # Normalize output by node degree.
        self.bias = None

    def forward(self, sample):

        edge_index = sample['bipar_edges'][0]
        x = sample['bipar_points'][0].double().clone().detach()

        pseudo = torch.rand((edge_index.size(1), 2), dtype=torch.double, requires_grad=False)  # two-dimensional edge attributes
        pseudo1 = torch.rand((edge_index.size(1), 2), dtype=torch.double, requires_grad=False)  # two-dimensional edge attributes


        # print(x.dtype)
        # print(pseudo.dtype)
        # print(self.weight.dtype)

        encode = spline_conv(x, edge_index, pseudo, self.weight, self.kernel_size,
                  self.is_open_spline, self.degree, self.norm, self.root_weight, self.bias)

        # print(encode.size())
        # print(self.weight1.size())
        
        decode = spline_conv(encode, edge_index, pseudo1, self.weight1, self.kernel_size,
                  self.is_open_spline, self.degree, self.norm, self.root_weight1, self.bias)
        
        return decode

    def encode(self, sample):
        """
        This is like forward function, but only to extract the intermediate sparse embeddings
        """
        edge_index = sample['bipar_edges'][0]
        x = sample['bipar_points'][0].double().clone().detach()

        pseudo = torch.rand((edge_index.size(1), 2), dtype=torch.double, requires_grad=False)  # two-dimensional edge attributes

        # print(x.dtype)
        # print(pseudo.dtype)
        # print(self.weight.dtype)

        encode = spline_conv(x, edge_index, pseudo, self.weight, self.kernel_size,
                  self.is_open_spline, self.degree, self.norm, self.root_weight, self.bias)

        return encode

       
    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'input_channel = {self.input_channel}, output_channel = {self.output_channel}, degree= {self.degree}, norm = {self.norm}\
            bias = {self.bias}, edge_dimension=2, num_kernel_per_dimension = {self.num_kernel}'

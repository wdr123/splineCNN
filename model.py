import torch
import math
import os
import numpy as np
from torch_spline_conv import spline_conv


class SplineConv(torch.nn.Module):
    def __init__(self, input_channel=6, embed_channel=1024, output_channel=3, num_kernel=5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.embed_channel = embed_channel
        self.num_kernel = num_kernel

        self.root_weight = torch.nn.Parameter(torch.rand((input_channel, embed_channel)))
        self.weight = torch.nn.Parameter(torch.rand((30, input_channel, embed_channel)))
        self.weight1 = torch.nn.Parameter(torch.rand((30, embed_channel, output_channel)))  # 25 parameters for in_channels x out_channels
        self.root_weight1 = torch.nn.Parameter(torch.rand((embed_channel, output_channel)))  # separately weight root nodes

        self.kernel_size = torch.tensor([num_kernel, num_kernel])  # 5 parameters in each edge dimension
        self.is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)  # only use open B-splines
        self.degree = 1  # B-spline degree of 1
        self.norm = True  # Normalize output by node degree.
        self.bias = None

    def forward(self, sample):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        edge_index = sample['edges'][0]
        x = sample['points'][0]

        pseudo = torch.rand((edge_index.size(1), 2), dtype=torch.float, requires_grad=False)  # two-dimensional edge attributes
        pseudo1 = torch.rand((edge_index.size(1), 2), dtype=torch.float, requires_grad=False)  # two-dimensional edge attributes

        encode = spline_conv(x, edge_index, pseudo, self.weight, self.kernel_size,
                  self.is_open_spline, self.degree, self.norm, self.root_weight, self.bias)

        # print(encode.size())
        # print(self.weight1.size())
        
        decode = spline_conv(encode, edge_index, pseudo1, self.weight1, self.kernel_size,
                  self.is_open_spline, self.degree, self.norm, self.root_weight1, self.bias)
        
        return decode
       
    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'input_channel = {self.input_channel}, output_channel = {self.output_channel}, degree= {self.degree}, norm = {self.norm}\
            bias = {self.bias}, edge_dimension=2, num_kernel_per_dimension = {self.num_kernel}'


# Create Tensors to hold input and outputs.
# x = torch.linspace(-math.pi, math.pi, 2000)
# y = torch.sin(x)

# Construct our model by instantiating the class defined above
# model = Polynomial3()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
# for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    # y_pred = model(x)

    # Compute and print loss
    # loss = criterion(y_pred, y)
    # if t % 100 == 99:
    #     print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# print(f'Result: {model.string()}')
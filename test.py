import torch
from torch_spline_conv import spline_conv

# x = torch.rand((4, 2), dtype=torch.float)  # 4 nodes with 2 features each
# edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])  # 6 edges
# pseudo = torch.rand((6, 2), dtype=torch.float)  # two-dimensional edge attributes
# weight = torch.rand((25, 2, 4), dtype=torch.float)  # 25 parameters for in_channels x out_channels
# kernel_size = torch.tensor([5, 5])  # 5 parameters in each edge dimension
# is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)  # only use open B-splines
# degree = 1  # B-spline degree of 1
# norm = True  # Normalize output by node degree.
# root_weight = torch.rand((2, 4), dtype=torch.float)  # separately weight root nodes
# bias = None  # do not apply an additional bias

encoder = spline_conv_6_1024(x, edge_index, pseudo, weight, kernel_size,
                  is_open_spline, degree, norm, root_weight, bias)

decoder = spline_conv_1024_3(x, edge_index, pseudo, weight, kernel_size,
                  is_open_spline, degree, norm, root_weight, bias)

mesh_covolute = spline_conv_1024_1024(x, edge_index, pseudo, weight, kernel_size,
                  is_open_spline, degree, norm, root_weight, bias)


x_mesh,edge_index,pseudo = loadmesh_input(dense = ['/home/dw7445/Projects/VU-summer/coarsen_dataset_m/pre.mat',\
                   '/home/dw7445/Projects/VU-summer/coarsen_dataset_m/deformed.mat'],
                   sparse = ['/home/dw7445/Projects/VU-summer/coarsen_dataset_m/sparse/srf.mat',\
                             '/home/dw7445/Projects/VU-summer/coarsen_dataset_m/sparse/posterior.mat',\
                            '/home/dw7445/Projects/VU-summer/coarsen_dataset_m/sparse/falciform.mat',\
                            '/home/dw7445/Projects/VU-summer/coarsen_dataset_m/sparse/leftRidge.mat',\
                            '/home/dw7445/Projects/VU-summer/coarsen_dataset_m/sparse/rightRdige.mat'])

y_mesh = loadmesh_supervise(dense = '/home/dw7445/Projects/VU-summer/coarsen_dataset_m/gt.mat')

'''
Args:
        x (:class:`Tensor`): Input node features of shape
            (number_of_nodes x in_channels).
        edge_index (:class:`LongTensor`): Graph edges, given by source and
            target indices, of shape (2 x number_of_edges) in the fixed
            interval [0, 1].
        pseudo (:class:`Tensor`): Edge attributes, ie. pseudo coordinates,
            of shape (number_of_edges x number_of_edge_attributes).
        weight (:class:`Tensor`): Trainable weight parameters of shape
            (kernel_size x in_channels x out_channels).
        kernel_size (:class:`LongTensor`): Number of trainable weight
            parameters in each edge dimension.
        is_open_spline (:class:`ByteTensor`): Whether to use open or closed
            B-spline bases for each dimension.
        degree (int, optional): B-spline basis degree. (default: :obj:`1`)
        norm (bool, optional): Whether to normalize output by node degree.
            (default: :obj:`True`)
        root_weight (:class:`Tensor`, optional): Additional shared trainable
            parameters for each feature of the root node of shape
            (in_channels x out_channels). (default: :obj:`None`)
        bias (:class:`Tensor`, optional): Optional bias of shape
            (out_channels). (default: :obj:`None`)'''
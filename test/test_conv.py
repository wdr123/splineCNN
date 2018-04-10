# import pytest
# import torch
# from torch.autograd import Variable, gradcheck
# from torch_spline_conv import spline_conv
# from torch_spline_conv.functions.spline_weighting import SplineWeighting
# from torch_spline_conv.functions.ffi import implemented_degrees

# from .utils import tensors, Tensor

# @pytest.mark.parametrize('tensor', tensors)
# def test_spline_conv_cpu(tensor):
#     x = Tensor(tensor, [[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
#     edge_index = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
#     pseudo = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
#     pseudo = Tensor(tensor, pseudo)
#     weight = torch.arange(0.5, 0.5 * 25, step=0.5, out=x.new()).view(12, 2, 1)
#     kernel_size = torch.LongTensor([3, 4])
#     is_open_spline = torch.ByteTensor([1, 0])
#     root_weight = torch.arange(12.5, 13.5, step=0.5, out=x.new()).view(2, 1)
#     bias = Tensor(tensor, [1])

#     output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
#                          is_open_spline, 1, root_weight, bias)

#     edgewise_output = [
#         1 * 0.25 * (0.5 + 1.5 + 4.5 + 5.5) + 2 * 0.25 * (1 + 2 + 5 + 6),
#         3 * 0.25 * (1.5 + 2.5 + 5.5 + 6.5) + 4 * 0.25 * (2 + 3 + 6 + 7),
#         5 * 0.25 * (6.5 + 7.5 + 10.5 + 11.5) + 6 * 0.25 * (7 + 8 + 11 + 12),
#         7 * 0.25 * (7.5 + 4.5 + 11.5 + 8.5) + 8 * 0.25 * (8 + 5 + 12 + 9),
#     ]

#     expected_output = [
#         [1 + 12.5 * 9 + 13 * 10 + sum(edgewise_output) / 4],
#         [1 + 12.5 * 1 + 13 * 2],
#         [1 + 12.5 * 3 + 13 * 4],
#         [1 + 12.5 * 5 + 13 * 6],
#         [1 + 12.5 * 7 + 13 * 8],
#     ]

#     assert output.tolist() == expected_output

#     x, weight, pseudo = Variable(x), Variable(weight), Variable(pseudo)
#     root_weight, bias = Variable(root_weight), Variable(bias)

#     output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
#                          is_open_spline, 1, root_weight, bias)

#     assert output.data.tolist() == expected_output

# def test_spline_weighting_backward_cpu():
#     for degree in implemented_degrees.keys():
#         kernel_size = torch.LongTensor([5, 5, 5])
#         is_open_spline = torch.ByteTensor([1, 0, 1])
#         op = SplineWeighting(kernel_size, is_open_spline, degree)

#         x = torch.DoubleTensor(16, 2).uniform_(-1, 1)
#         x = Variable(x, requires_grad=True)
#         pseudo = torch.DoubleTensor(16, 3).uniform_(0, 1)
#         pseudo = Variable(pseudo, requires_grad=True)
#         weight = torch.DoubleTensor(25, 2, 4).uniform_(-1, 1)
#         weight = Variable(weight, requires_grad=True)

#         assert gradcheck(op, (x, pseudo, weight), eps=1e-6, atol=1e-4) is True

# @pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
# @pytest.mark.parametrize('tensor', tensors)
# def test_spline_conv_gpu(tensor):  # pragma: no cover
#     x = Tensor(tensor, [[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
#     edge_index = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
#     pseudo = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
#     pseudo = Tensor(tensor, pseudo)
#     weight = torch.arange(0.5, 0.5 * 25, step=0.5, out=x.new()).view(12, 2, 1)
#     kernel_size = torch.LongTensor([3, 4])
#     is_open_spline = torch.ByteTensor([1, 0])
#     root_weight = torch.arange(12.5, 13.5, step=0.5, out=x.new()).view(2, 1)
#     bias = Tensor(tensor, [1])

#     expected_output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
#                                   is_open_spline, 1, root_weight, bias)

#     x, edge_index, pseudo = x.cuda(), edge_index.cuda(), pseudo.cuda()
#     weight, kernel_size = weight.cuda(), kernel_size.cuda()
#     is_open_spline, root_weight = is_open_spline.cuda(), root_weight.cuda()
#     bias = bias.cuda()

#     output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
#                          is_open_spline, 1, root_weight, bias)
#     assert output.cpu().tolist() == expected_output.tolist()

#     x, weight, pseudo = Variable(x), Variable(weight), Variable(pseudo)
#     root_weight, bias = Variable(root_weight), Variable(bias)

#     output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
#                          is_open_spline, 1, root_weight, bias)

#     assert output.data.cpu().tolist() == expected_output.tolist()

# @pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
# def test_spline_weighting_backward_gpu():  # pragma: no cover
#     for degree in implemented_degrees.keys():
#         kernel_size = torch.cuda.LongTensor([5, 5, 5])
#         is_open_spline = torch.cuda.ByteTensor([1, 0, 1])
#         op = SplineWeighting(kernel_size, is_open_spline, degree)

#         x = torch.cuda.DoubleTensor(16, 2).uniform_(-1, 1)
#         x = Variable(x, requires_grad=True)
#         pseudo = torch.cuda.DoubleTensor(16, 3).uniform_(0, 1)
#         pseudo = Variable(pseudo, requires_grad=False)  # TODO
#         weight = torch.cuda.DoubleTensor(25, 2, 4).uniform_(-1, 1)
#         weight = Variable(weight, requires_grad=True)

#         assert gradcheck(op, (x, pseudo, weight), eps=1e-6, atol=1e-4) is True
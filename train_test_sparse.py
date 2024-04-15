import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Use CUDA if available, otherwise error
assert torch.cuda.is_available()
device = torch.device("cpu")

from dataset import VUSparse

from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic

args = get_args()
make_deterministic(seed=args.seed)
print(args)

#Data Loaders
bs = 1
train_dl = DataLoader(VUSparse(root_dir="sparse_coarsen", ), batch_size=bs, shuffle=True)

# Import model
from sparse_model import SplineConv
model = SplineConv().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model_save_path = 'checkpoint_epoch_150.pth'
optim_save_path = 'optimizer_epoch_150.pth'


import csv
def save_to_csv(all_dicts, iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)

    save_log_name = f"Spline_{args.identifier}_Batch{args.batch_size}_sd{args.seed}.csv"
    # save_model_name = "Debug.csv"
    if iter == 0:
        with open(save_log_name, 'w+') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(header)
    with open(save_log_name, 'a') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(values)



def one_iteration_training(model, sample, label):
    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    decode = model(sample)

    loss = criterion(decode, label)
    
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy()



def main():
    for tr_it in range(150):
        # test_dict = test(args.identifier)

        loss_tr, count = 0.0, 0
        overall_count = 0

        for d in train_dl:
            # print(d['bipar_points'])
            # print(d['bipar_edges'])
            # print(d['register_name'])

            overall_count += 1
            label = d['bipar_points'][0]
            label = label.to(device).double()

            count += 1

            loss= one_iteration_training(model, d, label)
            loss_tr += loss

        print("overall count", overall_count)
        print("Tr Loss at it: ", tr_it, " loss: ", loss_tr / count)

        tr_loss = loss_tr / count
        tr_loss = tr_loss.item()

        train_dict = {
            'Train loss':tr_loss
        }

        all_dicts_list = [train_dict]

        save_to_csv(all_dicts_list, tr_it)

        # Print model's state_dict
        # print("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optim_save_path)

        # Print optimizer's state_dict
        # print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t", optimizer.state_dict()[var_name])


if __name__ == "__main__":
    main()

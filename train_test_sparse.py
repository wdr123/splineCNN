import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Use CUDA if available, otherwise error
assert torch.cuda.is_available()
device = torch.device("cpu")

from dataset import VUDataset

from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic

args = get_args()
make_deterministic(seed=args.seed)
print(args)

#Data Loaders
bs = 1
train_dl = DataLoader(VUDataset(root_dir="data_coarsen", seed = args.seed, train=True), batch_size=bs, shuffle=True)
# val_dl = DataLoader(VUDataset(root_dir="data_coarsen", args = args.seed, train=True), batch_size=1, shuffle=True)
test_dl = DataLoader(VUDataset(root_dir="data_coarsen", seed = args.seed, train=False), batch_size=bs, shuffle=True)

# Import model
from model import SplineConv
model = SplineConv().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


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

def test(identifier):
    if identifier == "train_test":
        the_dataloader = test_dl
    elif identifier == "val":
        print("NO validation now")
        raise NotImplementedError
        # the_dataloader = val_dl
    with torch.no_grad():
        model.eval()
        count,loss = 0, 0.0
        
        for d in the_dataloader:
            model.eval()
            model.zero_grad()
            optimizer.zero_grad()

            label = d['gt'][0]
            label = label.to(device).float()

            pred = model(d)

            # print("label: ", label)
            # print("prediction: ", pred)
            loss += criterion(pred, label)
            count += 1

    av_acc = loss/count
    av_acc = av_acc.detach().cpu().numpy().item()
    av_loss = loss/count
    av_loss = av_loss.detach().cpu().numpy().item()

    print("{} acc: ".format(identifier), av_acc, "count: ", count, 'loss: ', av_loss)

    the_dict = {
        identifier + " loss":av_loss,
        identifier + " acc": av_acc,
    }
    print('the_dict: ', the_dict)

    return the_dict


def one_iteration_training(model, sample, label):
    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    decode = model(sample)

    loss = criterion(decode, label)
    
    loss.backward()
    optimizer.step()

    accuracy = loss.detach().cpu().numpy()
    return loss.detach().cpu().numpy(), accuracy



def main():
    for tr_it in range(1000):
        test_dict = test(args.identifier)
        # val_dict = test("val")

        loss_tr, count = 0.0, 0
        accuracy = 0.0
        overall_count = 0

        for d in train_dl:
            overall_count += 1
            label = d['gt'][0]
            label = label.to(device).float()

            count += 1
            model.zero_grad()
            optimizer.zero_grad()
            model.train()

            loss, acc = one_iteration_training(model, d, label)
            loss_tr += loss
            accuracy += acc

        print("overall count", overall_count)
        print("Tr Loss at it: ", tr_it, " loss: ", loss_tr / count, "accuracy: ", accuracy / count)

        tr_loss = loss_tr / count
        tr_loss = tr_loss.item()
        tr_acc = accuracy / count
        tr_acc = tr_acc.item()

        train_dict = {
            'Train acc': tr_acc,
            'Train loss':tr_loss
        }

        all_dicts_list = [train_dict,test_dict]

        save_to_csv(all_dicts_list, tr_it)


if __name__ == "__main__":
    main()

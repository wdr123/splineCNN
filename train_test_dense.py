import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Use CUDA if available, otherwise error
assert torch.cuda.is_available()
device = torch.device("cpu")

from dataset import VUDense

from utilFiles.the_args import get_args
from utilFiles.set_deterministic import make_deterministic

args = get_args()
make_deterministic(seed=args.seed)
print(args)

# Hyper-parameter tuning the sparse loss comparing to the main dense loss
lambda_sparse = 0.5

#Data Loaders
bs = 1
train_dl = DataLoader(VUDense(root_dir="data_coarsen", seed = args.seed, train=True), batch_size=bs, shuffle=True)
test_dl = DataLoader(VUDense(root_dir="data_coarsen", seed = args.seed, train=False), batch_size=bs, shuffle=True)

# Import model
from model import SplineConv
model = SplineConv().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

best_model_save_path = 'checkpoint/checkpoint_best_dense.pth'
best_optim_save_path = 'checkpoint/optimizer_best_dense.pth'

last_model_save_path = 'checkpoint/checkpoint_epoch_200_dense.pth'
last_optim_save_path = 'checkpoint/optimizer_epoch_200_dense.pth'


import csv
def save_to_csv(all_dicts, iter=0):
    header, values = [], []
    for d in all_dicts:
        for k, v in d.items():
            header.append(k)
            values.append(v)

    save_log_name = f"Spline_{args.identifier}_Batch{args.batch_size}_sd{args.seed}_dense.csv"
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
        count,loss_dense, loss_sparse = 0, 0.0, 0.0
        
        for d in the_dataloader:
            model.eval()
            model.zero_grad()
            optimizer.zero_grad()

            label_dense = d['dense_gt'][0]
            label_sparse = d['sparse_supervision'][0]
            edge_ids = d['edge_points'][0]

            label_dense = label_dense.to(device).float()
            label_sparse = label_sparse.to(device).float()

            pred = model(d)

            # print("label_dense: ", label_dense)
            # print("prediction: ", pred)
            loss_dense += criterion(pred, label_dense)
            loss_sparse += criterion(pred[edge_ids], label_sparse)
            count += 1

    av_loss_dense = loss_dense/count
    av_loss_dense = av_loss_dense.detach().cpu().numpy().item()
    av_loss_sparse = loss_sparse/count
    av_loss_sparse = av_loss_sparse.detach().cpu().numpy().item()
    av_loss = av_loss_dense + lambda_sparse*av_loss_sparse

    print("{} loss_dense: ".format(identifier), av_loss_dense, "count: ", count, 'loss_sparse: ', av_loss_sparse)

    the_dict = {
        identifier + " loss_dense":av_loss_dense,
        identifier + " loss_sparse": av_loss_sparse,
        identifier + "loss": av_loss,
    }
    print('the_dict: ', the_dict)

    return the_dict, av_loss


def one_iteration_training(model, sample, label_dense, label_sparse):
    model.zero_grad()
    optimizer.zero_grad()
    model.train()

    pred = model(sample)

    # print("prediction: ", pred)
    loss_dense = criterion(pred, label_dense)
    # print("label_dense: ", label_dense)
    loss_sparse = criterion(pred[edge_ids], label_sparse)

    loss = loss_dense + lambda_sparse*loss_sparse 
    
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy()



def main():
    best_test_loss = np.Infinity

    for tr_it in range(200):
        test_dict, test_loss = test(args.identifier)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if os.path.exists(best_model_save_path):
                os.remove(best_model_save_path)
                os.remove(best_optim_save_path)
            torch.save(model.state_dict(), best_model_save_path)
            torch.save(optimizer.state_dict(), best_optim_save_path)

        loss_tr, count = 0.0, 0
        overall_count = 0

        for d in train_dl:
            overall_count += 1
            label_dense = d['dense_gt'][0]
            label_sparse = d['sparse_supervision'][0]
            edge_ids = d['edge_points'][0]

            label_dense = label_dense.to(device).float()
            label_sparse = label_sparse.to(device).float()

            count += 1
            model.zero_grad()
            optimizer.zero_grad()
            model.train()

            loss, acc = one_iteration_training(model, d, label_dense, label_sparse)
            loss_tr += loss

        print("overall count", overall_count)
        print("Tr Loss at it: ", tr_it, " loss: ", loss_tr / count, "accuracy: ", accuracy / count)

        tr_loss = loss_tr / count
        tr_loss = tr_loss.item()


        train_dict = {
            'Train loss':tr_loss
        }

        all_dicts_list = [train_dict,test_dict]

        save_to_csv(all_dicts_list, tr_it)

   
    torch.save(model.state_dict(), last_model_save_path)
    torch.save(optimizer.state_dict(), last_optim_save_path)


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import torch.nn as nn
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

sparse_dir = "sparse_coarsen"

#Data Loaders
bs = 1
test_dl = DataLoader(VUSparse(root_dir="sparse_coarsen", ), batch_size=bs, shuffle=True)
criterion = nn.MSELoss()

# Import model
from sparse_model import SplineConv
model = SplineConv().to(device)
ckptPath = 'checkpoint_epoch_150.pth'
model.load_state_dict(torch.load(ckptPath))
model.eval()


def test(identifier):
    if identifier == "train_test":
        the_dataloader = test_dl
    elif identifier == "val":
        print("NO validation now")
        raise NotImplementedError
        # the_dataloader = val_dl
    with torch.no_grad():
        count,loss = 0, 0.0
        
        for d in the_dataloader:

            d['register_name'] = d['register_name'][0]
            save_path = os.path.join(sparse_dir,d['register_name'][:3],d['register_name'], 'embedding.npy')
            label = d['bipar_points'][0]
            label = label.to(device).double()

            pred = model(d)
            sparse_encoding = model.encode(d)
            if os.path.exists(save_path):
                os.remove(save_path)
                torch.save(sparse_encoding, save_path)

            # print("label: ", label)
            # print("prediction: ", pred)
            loss += criterion(pred, label)
            count += 1


    av_loss = loss/count
    av_loss = av_loss.detach().cpu().numpy().item()

    print("count: ", count, 'loss: ', av_loss)

    the_dict = {
        identifier + " loss":av_loss,
    }
    print('the_dict: ', the_dict)

    return the_dict


if __name__ == "__main__":
    test(args.identifier)







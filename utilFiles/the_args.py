import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU index used for the code")
    parser.add_argument('--seed', type = int, default=18,help="Seed for the code")
    parser.add_argument('--lr', type=int, default=0.0001, help="Optimizer learning rate")
    parser.add_argument('--weight_decay', type=int, default=1e-5, help="Learning rate weight decay")
    parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
    parser.add_argument('-id','--identifier', type=str, default="train_test", help="Identifier For Script")
    # parser.add_argument('--model', choices=['combine', 'no_attention', 'attention_only'], default='combine', help="model structure")


    args = parser.parse_args()

    return args
from scipy.io import loadmat, savemat
import os

# df = loadmat('dataset_mat_correct/1-L/PreOperative/pre_points.mat')

# print(df)

for dirpath, dirnames, filenames in os.walk('dataset_mat_correct'):
    if dirpath == 'dataset_mat_correct/1-L':
        print('1-L', len(dirnames)-1)
    elif dirpath == 'dataset_mat_correct/1-R':
        print('1-R', len(dirnames)-1)
    elif dirpath == 'dataset_mat_correct/2-L':
        print('2-L', len(dirnames)-1)
    elif dirpath == 'dataset_mat_correct/2-R':
        print('2-R', len(dirnames)-1)
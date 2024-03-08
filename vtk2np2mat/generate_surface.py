import os
import numpy as np
from convertVtk2Np_Poly import convertVtk2Np_Poly
from convertVtk2Np_UG import convertVtk2Np_UG
from scipy.io import savemat
import glob

filename_list = ['01242014_Posterior_wICPRegistered', '01242014_srf_wICPRegistered', 
                 '02062011_Posterior_wICPRegistered', '02062011_srf_wICPRegistered']



def np2mat(srcNumpyArray, dstFilePath):
    dstFileDir = '/'.join(dstFilePath.split('/')[:-1])

    if os.path.exists(os.path.splitext(dstFilePath)[0]+'.mat'):
        return
    
    if not os.path.exists(dstFileDir):
         os.makedirs(dstFileDir)
    
    np.savez(dstFilePath, srcNumpyArray)
    
    searchPath = os.path.join(dstFileDir, "*.npz")

    npzFiles = glob.glob(searchPath)
    for f in npzFiles:
        fm = os.path.splitext(f)[0]+'.mat'
        d = np.load(f)
        savemat(fm, d)
        print('generated ', fm, 'from', f)
        os.remove(f)



for root in ['raw/1-L', 'raw/1-R', 'raw/2-L', 'raw/2-R']:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for filename in filenames:
            if filename.split('.')[0] in filename_list:
                filepath = os.path.join(dirpath, filename)                
                if filename.split('.')[0].split('_')[-2] == 'Posterior':
                    points = convertVtk2Np_Poly(filepath)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('posterior_points')
                    dstPointsFilePath = '/'.join(filenameList)
                    np2mat(points, dstPointsFilePath)

                elif filename.split('.')[0].split('_')[-2] == 'srf':
                    points = convertVtk2Np_Poly(filepath)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('srf_points')
                    dstPointsFilePath = '/'.join(filenameList)
                    np2mat(points, dstPointsFilePath)



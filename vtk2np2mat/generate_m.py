import os
import numpy as np
from convertVtk2Np_Poly import convertVtk2Np_Poly
from convertVtk2Np_UG import convertVtk2Np_UG
from scipy.io import savemat
import glob

filename_list = ['0124_mesh', '0124_meshtre_IntraOp_wICPRegistered', '01242014_mesh_deformed_smoothed', 
                 '0206_mesh', '0206_meshtre_IntraOp_wICPRegistered', '02062011_mesh_deformed_smoothed']



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
                if filename.split('.')[0].split('_')[-1] == 'wICPRegistered':
                    points = convertVtk2Np_Poly(filepath)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('gt_points')
                    dstPointsFilePath = '/'.join(filenameList)
                    np2mat(points, dstPointsFilePath)

                elif filename.split('.')[0].split('_')[-1] == 'mesh':
                    points, cells = convertVtk2Np_UG(filepath)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('pre_points')
                    dstPointsFilePath = '/'.join(filenameList)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('pre_cells')
                    dstCellsFilePath = '/'.join(filenameList)
                    np2mat(points, dstPointsFilePath)
                    np2mat(cells, dstCellsFilePath)
                    
                elif filename.split('.')[0].split('_')[-1] == 'smoothed':
                    points, cells = convertVtk2Np_UG(filepath)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('deformed_points')
                    dstPointsFilePath = '/'.join(filenameList)
                    filenameList = filepath.replace('raw','dataset_mat').split('/')[:-1]
                    filenameList.append('deformed_cells')
                    dstCellsFilePath = '/'.join(filenameList)
                    np2mat(points, dstPointsFilePath)
                    np2mat(cells, dstCellsFilePath)     



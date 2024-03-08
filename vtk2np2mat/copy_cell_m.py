import numpy as np
import os
import shutil


def copy_cell_m(root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for dirname in dirnames:
            if dirname == "IntraOperative":
                srcfilepath = os.path.join(dirpath, dirname, 'deformed_cells.mat')
                dstfilepath = os.path.join(dirpath, 'gt_cells.mat')
                if not os.path.exists(dstfilepath):
                    shutil.copy(srcfilepath, dstfilepath)


copy_cell_m('dataset_mat')
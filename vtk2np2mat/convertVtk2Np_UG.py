import numpy as np
from vtk import vtkUnstructuredGridReader, vtkPoints, vtkUnstructuredGridWriter
import os
from vtk.numpy_interface import dataset_adapter as dsa




def convertVtk2Np_UG(vtkFilePath):
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(vtkFilePath)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    # data = reader.GetOutput()
    unstructuredGrid = reader.GetOutput()

    npPoints = dsa.WrapDataObject(unstructuredGrid).Points
    npCells = dsa.WrapDataObject(unstructuredGrid).Cells

    cells = npCells.reshape(-1, 5)[:,1:]
    points = npPoints * 1000

    return points, cells







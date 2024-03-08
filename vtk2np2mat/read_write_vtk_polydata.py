import numpy
from vtk import vtkUnstructuredGridReader, vtkPolyDataReader, vtkPoints, vtkUnstructuredGridWriter, vtkPolyDataWriter
from vtk.util import numpy_support as VN

reader = vtkPolyDataReader()
reader.SetFileName('ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_bel_deformed_smoothed.vtk')
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()

# data = reader.GetOutput()

from vtk.numpy_interface import dataset_adapter as dsa

PolyData = reader.GetOutput()
numpy_array_of_points = dsa.WrapDataObject(PolyData).Points
number_of_points = len(numpy_array_of_points)

points = vtkPoints()
for id in range(number_of_points):
    points.InsertPoint(id, numpy_array_of_points[id]*1000)
PolyData.SetPoints(points)

# numpy_array_of_points = dsa.WrapDataObject(unstructuredGrid).Points
# print(numpy_array_of_points)

writer = vtkPolyDataWriter()
writer.SetFileName("ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_bel_deformed_smoothed_back.vtk")
writer.SetInputData(PolyData)
writer.Write()



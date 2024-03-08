import numpy
from vtk import vtkUnstructuredGridReader, vtkPolyDataReader, vtkPoints, vtkUnstructuredGridWriter
from vtk.util import numpy_support as VN

reader = vtkUnstructuredGridReader()
reader.SetFileName('ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_mesh_deformed_smoothed.vtk')
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()

# data = reader.GetOutput()

from vtk.numpy_interface import dataset_adapter as dsa

unstructuredGrid = reader.GetOutput()
numpy_array_of_points = dsa.WrapDataObject(unstructuredGrid).Points
number_of_points = len(numpy_array_of_points)
print(numpy_array_of_points.shape)




# points = vtkPoints()
# for id in range(number_of_points):
#     points.InsertPoint(id, numpy_array_of_points[id]*1000)
# unstructuredGrid.SetPoints(points)


# writer = vtkUnstructuredGridWriter()
# writer.SetFileName("ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_mesh_deformed_smoothed_back.vtk")
# writer.SetInputData(unstructuredGrid)
# writer.Write()


# np.savez('deformed_mesh_tet', cells)

# npzFiles = glob.glob("*.npz")
# for f in npzFiles:
#     fm = os.path.splitext(f)[0]+'.mat'
#     d = np.load(f)
#     savemat(fm, d)
#     print('generated ', fm, 'from', f)



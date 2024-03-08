import pyvista as pv
import os


class ShowVTK():
    def __init__(self):
        self.plotter = None

    def load_vtk(self, f, c):
        mesh = pv.read(f)
        self.plotter.add_mesh(mesh, color=c, opacity=1.0,
                              show_edges=True, line_width=0.5)

    def load_mesh(self, f, c):
        mesh = pv.read(f)
        self.plotter.add_mesh(mesh, color=c, opacity=1.0, point_size=2)

    def call_back(self):
        print("test")
        print(self.plotter.camera_position)

    def main(self, p, name):
        path = os.path.join(p, name)
        # Create a PyVista plotter

        self.plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
        self.load_mesh(
            'VU_summer/ExampleDataDingrong/0124-LeftMob_0913-USPlane0/0124_meshtre_IntraOp_wICPRegistered.vtk', 'white')
        self.load_mesh("/Users/soheil/Downloads/Mesh_coarsen_visualization_toolset/VU_summer/ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_Falciform_wICPRegistered.vtk", 'yellow')
        self.load_mesh("/Users/soheil/Downloads/Mesh_coarsen_visualization_toolset/VU_summer/ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_LeftInferiorRidge_wICPRegistered.vtk", 'yellow')
        self.load_mesh("/Users/soheil/Downloads/Mesh_coarsen_visualization_toolset/VU_summer/ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_RightInferiorRidge_wICPRegistered.vtk", 'yellow')

        # self.load_vtk(os.path.join(path, "deformed_mesh.vtk"), "red")
        # self.load_vtk(os.path.join(path, "pre_mesh.vtk"), "blue")
        self.load_vtk("/Users/soheil/Downloads/Mesh_coarsen_visualization_toolset/VU_summer/ExampleDataDingrong/0124-LeftMob_0913-USPlane0/PreOperative/0124_mesh_align.vtk", "blue")
        self.load_vtk("/Users/soheil/Downloads/Mesh_coarsen_visualization_toolset/VU_summer/ExampleDataDingrong/0124-LeftMob_0913-USPlane0/IntraOperative/01242014_mesh_deformed_smoothed_align.vtk", "red")
        self.plotter.background_color = 'black'
        self.plotter.enable_3_lights()
        self.plotter.isometric_view()
        self.plotter.set_viewup([0, 1, 0])
        # self.plotter.camera_position = "XZ"
        # plotter.save_graphic(f"vtk/{name}.pdf")
        print(name)
        # plotter.show()
        # self.plotter.camera_position = [(-323.1965353213811, 252.96148910325346, 186.20759706096038),
        #                                 (140.95575714111328, 211.12714767456055,
        #                                  160.49999618530273),
        #                                 # (0.07164486379297627, 0.1936302816156865, 0.9784550718012127)]
        #                                 (0.0, 0.0, 1.0)]
        self.plotter.camera_position = "XZ"
        self.plotter.screenshot(f"vtk/{name}_XZ.png")
        self.plotter.camera_position = "ZX"
        self.plotter.screenshot(f"vtk/{name}_ZX.png")

        # self.plotter.camera_position = [(363.2668321551898, 308.4630887765792, -238.1876559430424),
        #                                 (140.95575714111328, 211.12714767456055,
        #                                  160.49999618530273),
        #                                 (0.513447729741781, 0.7226672895578463, 0.4627347160373068)]
        self.plotter.camera_position = "YX"
        self.plotter.screenshot(f"vtk/{name}_YX.png")
        self.plotter.camera_position = "XY"
        self.plotter.screenshot(f"vtk/{name}_XY.png")

        # self.plotter.camera_position = [(-94.45438191642516, -22.4570246406674, 488.93392308151044),
        #                                 (140.95575714111328, 211.12714767456055,
        #                                  160.49999618530273),
        #                                 (0.4651245032483415, 0.5291292817036725, 0.7097051498486815)]
        self.plotter.camera_position = "YZ"
        self.plotter.screenshot(f"vtk/{name}_YZ.png")
        self.plotter.camera_position = "ZY"
        self.plotter.screenshot(f"vtk/{name}_ZY.png")

        # self.plotter.add_key_event('d', self.call_back)
        # self.plotter.show()


if __name__ == '__main__':
    show = ShowVTK()
    # You can set this folder to the dataset_compare paths
    path = "/Users/soheil/Downloads/Mesh_coarsen_visualization_toolset/VU_summer/dataset_compare/"
    listdir = os.listdir(path)
    print(listdir)
    isTest = False
    # isTest = True
    if isTest:
        show.main(path, "1-L_01242014-USPlane0")
    else:
        for p in listdir:
            if os.path.isdir(os.path.join(path, p)):
                show.main(path, p)

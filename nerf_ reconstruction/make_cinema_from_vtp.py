import pyvista as pv

from nerf_imdb.cinema_exporter import cinema_rgba_image_exporter
from nerf_imdb.fibonacci_lattice import fibonacci_lattice
import sys
import numpy as np

if len(sys.argv) > 2:
    file = sys.argv[1]
    output = sys.argv[2]
else:
    print("Enter file path: ")
    sys.exit()


if __name__ == '__main__':
    data = pv.read(file)
#    scalarName = data.GetPointData().GetScalars().GetName()
#    range = data.GetPointData().GetScalars().GetRange()
#    print('scalarName', scalarName, (range[0]+range[1])/2)
    
#    dims = data.dimensions

#    contour = data.contour(scalars=scalarName, isosurfaces=[(range[0]+range[1])/2])

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(data, opacity=1.0, cmap='coolwarm')
    # Create a cube from [-0.5, 0.5]^3
    cube = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=1.0, z_length=1.0)
    plotter.add_mesh(cube, style="wireframe", color="black", line_width=2, opacity=0.0)
#    plotter.add_volume(data, opacity=0.0, cmap=['white'])

#    plotter.remove_scalar_bar(scalarName)
    plotter.background_color = 'black'
    plotter.camera.enable_parallel_projection()
#    plotter.window_size = [512, 512]
    plotter.window_size = [256, 256]
#    plotter.window_size = [128, 128]
    plotter.reset_camera()

    centerElevation = [90]
    centerAzimuth = [0]
##    centerElevation = [0]
##    centerAzimuth = [0]
##    centerElevation, centerAzimuth = fibonacci_lattice(100)
#    # 6 images for tilting your haed
##    elevations = [centerElevation[0]-10, centerElevation[0], centerElevation[0]+10, centerElevation[0]-10, centerElevation[0], centerElevation[0]+10]
##    azimuths = [centerAzimuth[0]-10, centerAzimuth[0]-10, centerAzimuth[0]-10, centerAzimuth[0]+10, centerAzimuth[0]+10, centerAzimuth[0]+10]
    # 0 images for tilting your haed
    offset = 10
    elevations = [centerElevation[0]-offset, centerElevation[0], centerElevation[0]+offset, centerElevation[0]-offset, centerElevation[0], centerElevation[0]+offset,centerElevation[0]-offset, centerElevation[0], centerElevation[0]+offset]
    azimuths = [centerAzimuth[0]-offset, centerAzimuth[0]-offset, centerAzimuth[0]-offset, centerAzimuth[0], centerAzimuth[0], centerAzimuth[0], centerAzimuth[0]+offset, centerAzimuth[0]+offset, centerAzimuth[0]+offset]
    elevations = [((e + 90) % 180) - 90 for e in elevations]
    azimuths = [a % 360 for a in azimuths]
##    # 2 images for stereoscopic view
##    elevations = [centerElevation, centerElevation]
##    azimuths = [centerAzimuth-10, centerAzimuth+10]
##    for i in range(len(centerElevation)):
##        # 1 imge for gt
##        elevations = [centerElevation[i]]
##        azimuths = [centerAzimuth[i]]
#    elevations, azimuths = fibonacci_lattice(250)
##    elevations = np.linspace(-90,90,3)
##    azimuths = np.linspace(-90,90,3)
##    elevations = np.zeros(3)
##    elevations = azimuths = [0]
##    elevations = [45,45,45]
##    elevations = azimuths = [35, 45, 55] # 3
##    elevations = [35,45,55, 35,45,55, 35,45,55] # 9
##    azimuths = [35, 35, 35, 45, 45, 45, 55, 55, 55]
##    elevations = [35,45,55, 35,45,55] # 6
##    azimuths = [35, 35, 35, 55, 55, 55]
#    elevations = [45]
#    azimuths = [45]
##    print(elevations, azimuths)
    cinema_rgba_image_exporter(plotter, elevations, azimuths, output)

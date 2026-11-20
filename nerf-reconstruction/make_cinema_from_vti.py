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
    scalarName = data.GetPointData().GetScalars().GetName()
    range = data.GetPointData().GetScalars().GetRange()
    print('scalarName', scalarName, (range[0]+range[1])/2)
    
    dims = data.dimensions

    contour = data.contour(scalars=scalarName, isosurfaces=[(range[0]+range[1])/2])

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(contour, scalars=scalarName, opacity=1.0, cmap="rainbow", clim=[range[0], range[1]])
    plotter.add_volume(data, opacity=0, cmap=['black'])

    plotter.remove_scalar_bar(scalarName)
    plotter.background_color = 'black'
    plotter.camera.enable_parallel_projection()
#    plotter.window_size = [256, 256]
    plotter.window_size = [128, 128]
    plotter.reset_camera()
#    print(plotter.camera)
#    sys.exit()
    

#    elevations, azimuths = fibonacci_lattice(250)
    centerElevation = [0]
    centerAzimuth = [90]
    # 9 images for tilting your haed
    offset = 10
    elevations = [centerElevation[0]-offset, centerElevation[0], centerElevation[0]+offset, centerElevation[0]-offset, centerElevation[0], centerElevation[0]+offset,centerElevation[0]-offset, centerElevation[0], centerElevation[0]+offset]
    azimuths = [centerAzimuth[0]-offset, centerAzimuth[0]-offset, centerAzimuth[0]-offset, centerAzimuth[0], centerAzimuth[0], centerAzimuth[0], centerAzimuth[0]+offset, centerAzimuth[0]+offset, centerAzimuth[0]+offset]
    elevations = [((e + 90) % 180) - 90 for e in elevations]
    azimuths = [a % 360 for a in azimuths]
    print(elevations, azimuths)
    cinema_rgba_image_exporter(plotter, elevations, azimuths, output)

import numpy as np
import vtk
from vtk.util import numpy_support
import os
import re
import glob
import cv2
import open3d as o3d


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


width, height = 512, 512
num_slices = 1
spacing = 1
img_list = []
  
index = 0
for img in sorted(glob.glob('/home/haitham/Desktop/25-10-2024/GRE/output_after_mapping' + '/*.png'),key=numericalSort):
  img_list.append(np.load(img))
  index += 1

combined_image_data = np.concatenate(img_list, axis=0)


vtk_data = numpy_support.numpy_to_vtk(combined_image_data.reshape(-1, 3), deep=True)
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(width, height, len(img_list))
vtk_image.SetSpacing(1.0, 1.0, spacing)
# since we are loading the vtk I am not sure if we can change the spacing as you do in paraview but you can set the spacing here instead.
vtk_image.GetPointData().SetScalars(vtk_data)
xx=vtk_image.GetPointData().GetScalars()
vtk_data.SetName('Channels_and_opacity')

# Write the VTK image data to a .vti file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(r'/home/haitham/Desktop/25-10-2024/GRE/corrected_GRE.vti')
writer.SetInputData(vtk_image)
writer.Write()

# Confirmation message
print("Image saved as image.vti")
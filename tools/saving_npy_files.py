import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
from tifffile import imread
import tifffile as tiff
import glob
import re
import time

start_time = time.time()
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def append_images_to_tiff(tiff_file, images):
    with tiff.TiffWriter(tiff_file, append=True) as tiff_writer:
        for image in images:
            tiff_writer.write(image, contiguous=True)
            
def To_npy(integral_path, dof_path, out_dir, w, h):
    
    imgs = sorted(glob.glob(integral_path + '/*.png'),key=numericalSort)
    dof_images = sorted(glob.glob(dof_path + '/*.tif'),key=numericalSort)
    
    for index, i in enumerate(imgs):
        
        print(os.path.join(integral_path, i))
        IMAGE_DATA = cv2.imread(os.path.join(integral_path, str(imgs[index])), 0) 
        DOF = imread(os.path.join(dof_path, str(dof_images[index]))) # depth from focus data using any tool (resnet, variance, etc ...) 

        alpha_channel = np.ones((h, w), dtype= np.float32) # 440 x 440
        alpha_channel = DOF  
                  
        alpha_channel = (alpha_channel - (np.min(alpha_channel)) ) / ((np.max(alpha_channel)) - (np.min(alpha_channel)))
        new_image_data = np.stack((IMAGE_DATA, IMAGE_DATA, alpha_channel), axis= -1)      
        np.save(os.path.join(out_dir, str(i.split('.')[0].split('\\')[-1])+'.npy'), new_image_data)
        
if __name__ == '__main__':
    integral_path = r'path'
    resnet_path = r'path'
    out_dir = r'path'
    w, h = 440, 440
    To_npy(integral_path, resnet_path, out_dir,w, h)
    
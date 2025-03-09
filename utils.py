from slice_contribution import slice_contribution
import os
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image, ImageOps
import pickle
import time 
from tools.utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed

#####################################################################################
## This File generates the training data for all pixels in Focal stack 440x440x440 ##
#####################################################################################
def process_pixel(x, y, z, save_path, raw_data_path, axil):   

    ground_truth = os.path.join('data/raw_data', raw_data_path, 'ZS_cropped')
    slices = os.path.join('data/raw_data', raw_data_path, 'FP_cropped')


    ground_truth_img = Image.open(os.path.join(sorted(glob.glob(ground_truth + '/*.png'),key=numericalSort)[z - 1])).convert('L')
    # print(os.path.join(sorted(glob.glob(ground_truth + '/*.png'),key=numericalSort)[z - 1]))
    pixel = ground_truth_img.getpixel((x, y))
    extra_zeros = np.zeros((2, 2))

    if pixel != 0:
        new_axil = 1 if axil == 0 else axil
        # x, y, z, #of_slices, path_of_images, axil skipping resolution 10 means (extract layer every 10 layers)
        mona = slice_contribution(x, y, z, 440, ground_truth, 1, True, new_axil)
        z_vector = [split_image_into_equal_tiles(value, 2) for value in mona]

        mona = slice_contribution(x, y, z, 440, slices, 0, True, new_axil)
        v = [split_image_into_equal_tiles(value, 2) for value in mona]

        if axil == 0:
            for i in range(20 - (abs(440 - z))):
                v.append(extra_zeros)
        try:
            dict_data = {}
            dict_data['input'] = np.array(v)
            dict_data['target'] = np.array(z_vector)
            pickle_name = str(x) + '_' + str(y) + '_' + str(z) + "_"+raw_data_path
            with open(os.path.join(save_path, pickle_name +'.pkl'), 'wb') as f:
                pickle.dump(dict_data, f)  
        except:
           print('error')



def parallel_process_image(x_range, y_range, z, save_path, raw_data_path, axil):
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=2) as executor:
    # with ProcessPoolExecutor() as executor:
        futures = []
        # Submit tasks to the executor
        for x in tqdm(x_range, desc="Processing x"):
            for y in y_range:
                futures.append(executor.submit(process_pixel, x, y, z, save_path, raw_data_path, axil))
    
        # Progress bar and wait for all tasks to complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing y", leave=False):
            try:
                future.result()  # Collect the result
            except Exception as e:
                print(f"Error in processing: {e}")

# Ensure this block is only executed when the script is run directly
if __name__ == '__main__':


    x_range = range(0, 440)  # Adjust as necessary
    y_range = range(0, 440)  # Adjust as necessary
    z_range = range(1, 441)
    fodler_path = r'data/train'
    
    DIR = 'data/raw_data'
    
    FOLDERS = os.listdir(DIR)
    print(FOLDERS)
    for folder in FOLDERS[3:]:
        for z in z_range:
            if not os.path.exists(os.path.join(fodler_path, 'Layer_'+str(z))):
                os.makedirs(os.path.join(fodler_path, 'Layer_'+str(z)))

            file_path = "layers_dataa.txt"
            content = f'Layer Number = {z} , axil resolution is {int(abs(z - 440) // 20)}, {folder} \n'
            with open(file_path, "a") as file:
                file.write(content)
            # Assuming you have your ground_truth_img loaded here
            parallel_process_image(x_range, y_range, z, os.path.join(fodler_path, 'Layer_'+str(z)), folder, axil=int(abs(z - 440) // 20))

            print(f'--------------------- Layer number {z}, Folder name {folder} is finished ---------------------')

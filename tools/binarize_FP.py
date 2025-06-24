from PIL import Image
import numpy as np
import os


import re

def natural_key(string):
    # Use regex to split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]


GROUND_PATH = r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_222223\ZS_cropped'
FS_PATH = r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_222223\FP_cropped'

imgs = os.listdir(GROUND_PATH)

imgs = sorted(imgs, key=natural_key)



for index, img in enumerate(imgs):
    image_num = img.split('TD')[-1].split('_')[0]
    
    # Load the images
    gt = os.path.join(GROUND_PATH, img)  # Replace with your TIFF image path
    resnet = os.path.join(FS_PATH, image_num + '.png')

    gt = Image.open(gt).convert('L')  # Convert to grayscale
    resnet = Image.open(resnet).convert('L')  # Convert to grayscale

    # Convert images to numpy arrays
    gt_array = np.array(gt)
    resnet_array = np.array(resnet)

    width, height = resnet.size

    non_zero_pixels = []

    for y in range(height):
        for x in range(width):
            pixel_value = gt.getpixel((x, y))
            if pixel_value:
                non_zero_pixels.append(((x, y), pixel_value))
                p_v = resnet.getpixel((x, y))
                resnet_array[y, x] = p_v
            else:
                resnet_array[y, x] = 0

    mona = Image.fromarray(resnet_array, mode='L')
    mona.save(os.path.join(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_222223\FP_cropped_binary', str(index + 1) + '.png'))
    true_positive , false_positive = 0, 0

    # for i in non_zero_pixels:
    #     if gt_array[i[0][1], i[0][0]]:
    #         true_positive += 1
    #     else:
    #         false_positive += 1

    # print(true_positive, false_positive)


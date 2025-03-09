import torch
from tqdm import tqdm
import os
from CNN3D.cnn_model import Simple3DCNN
from slice_contribution import slice_contribution
import numpy as np
from PIL import Image
import glob
from tools.utils import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor,  as_completed
import pickle
from pkl_handler import PKLFileHandler
import multiprocessing

def test_parallel(x, y, model, ground_truth_img, random_matrix, device, model_axil, layer):

    z = layer
    extra_zeros = np.zeros((2, 2))

    psf = slice_contribution(x, y, z, 440, 'slices', 0, False, round((440 - z) / model_axil)+1)
    v = [split_image_into_equal_tiles(value, 2) for value in psf]

    # if True:
    #     for i in range(20 - (abs(z - 440))):
    #         v.append(v[0])

    input_data = np.array(v)

    with torch.no_grad():
        inp = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)
        test_outputs = model(inp.permute(1, 0, 2, 3, 4).float().to(device))

    output_value = test_outputs.item() * 255
    return (x, y, output_value)  # Return coordinates along with the value

def process_pixel(pixel_info, model_save_path, ground_truth_img, random_matrix, device, model_axil, layer):

    # Load the model inside the process
    model = Simple3DCNN(in_channelss=model_axil).cuda()
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.to(device)
    model.eval()

    x, y = pixel_info[0]
    return test_parallel(x, y, model, ground_truth_img, random_matrix, device, model_axil, layer)

def main(layer, model_axil):
    ground_truth = r'data/test/Scene_507/ZS_cropped'
    layer = int(layer)
    ground_truth_img = Image.open(os.path.join(sorted(glob.glob(ground_truth + '/*.png'), key=numericalSort)[layer - 1])).convert('L')
    print(os.path.join(sorted(glob.glob(ground_truth + '/*.png'), key=numericalSort)[layer - 1]), layer)
    
    empty_image = Image.new('L', (440, 440), color=(0))
    random_matrix = np.zeros_like(empty_image)

    model_save_path = f'checkpoint/Layer_40.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    non_zero_pixels = []
    width, height = ground_truth_img.size

    # Collect all non-zero pixels
    for y in range(height):
        for x in range(width):
            pixel_value = ground_truth_img.getpixel((x, y))
            if pixel_value != 0:
            #if True:
                non_zero_pixels.append(((x, y), pixel_value))

    # Parallel processing of non-zero pixels using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_pixel, i, model_save_path, ground_truth_img, random_matrix, device, model_axil, layer) for i in non_zero_pixels]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="item"):
            try:
                x, y, output_value = future.result()
                random_matrix[y, x] = output_value
            except Exception as exc:
                print(f'Error occurred: {exc}')

    # Save the resulting image
    image = Image.fromarray(random_matrix, mode='L')
    image.save("outputs\Layer_"+str(layer)+"_new.png")
    image.close()

def load_image_stack(directory, image_type, layer_number, axil = 1):
    # Pattern to match the desired image files
    pattern = os.path.join(directory, '*.png')
    imgs = []
    # Sorted list of matching files
    file_list = sorted(glob.glob(pattern), key=numericalSort)
    file_list = file_list[layer_number - 1::axil]

    for filename in file_list:
        # Load the image and convert to grayscale ('L')
        img = Image.open(filename).convert('L')
        image_array = np.array(img)
        imgs.append(image_array)

    return np.array(imgs)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    with open('layers_data.txt', "r") as file:
        lines = file.readlines()

    for layer in range(1, 441):
        main(layer, int(lines[layer - 1].split()[-1]))

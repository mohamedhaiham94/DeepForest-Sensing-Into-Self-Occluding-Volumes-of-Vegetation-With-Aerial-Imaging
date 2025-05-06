from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
imgs = os.listdir(r'd:\Research\De-Blurring\Real Data Recordings\Top Down\F_Day\0063\RGB_solved')
index = 0
imgs.sort()
print(imgs)

for img in imgs[5:]:
    num = img.split('.')[0]

    
    rgb_img2 = Image.open(os.path.join(r'd:\Research\De-Blurring\Real Data Recordings\Top Down\F_Day\0063\RGB_solved', img))

    width, height = rgb_img2.size

    # # Set the desired crop size
    crop_width, crop_height = 4608, 3456  # Adjust as needed

       
    start_x2 = ((width - crop_width) // 2)
    start_y2 = ((height - crop_height) // 2)
    
    # Perform the crop
    cropped_rgb_image2 = rgb_img2.crop((10, 380, rgb_img2.width - 630, rgb_img2.height - 100))

    cropped_rgb_image2.save(os.path.join(r'd:\Research\De-Blurring\Real Data Recordings\Top Down\F_Day\0063\RGB_cropped', img))
    index += 1


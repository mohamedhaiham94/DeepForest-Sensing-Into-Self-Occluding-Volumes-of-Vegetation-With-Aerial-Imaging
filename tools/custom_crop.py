from PIL import Image
import os
from PIL import ImageFile
import piexif


ImageFile.LOAD_TRUNCATED_IMAGES = True
imgs = os.listdir(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_504_420_T\ZS')
index = 0
imgs.sort()
print(imgs)

for img in imgs:
    num = img.split('_')[0].split('D')[-1]
    print(num)
    

    
    rgb_img2 = Image.open(os.path.join(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_504_420_T\FP', str(num) + '.png'))

    width, height = rgb_img2.size

    crop_width, crop_height = 440, 440  # Adjust as needed
    
    # Crop the image
    cropped_rgb_image2 = rgb_img2.crop((260, 260, rgb_img2.width - 260, rgb_img2.height - 260))
    
    # Left Up Right Down

    cropped_rgb_image2.save(os.path.join(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_504_420_T\FP_cropped', str(num) + '.png'))
    index += 1


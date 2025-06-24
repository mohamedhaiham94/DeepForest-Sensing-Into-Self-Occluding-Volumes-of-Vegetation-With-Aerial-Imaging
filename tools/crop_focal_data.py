from PIL import Image
import os
from PIL import ImageFile
import piexif


ImageFile.LOAD_TRUNCATED_IMAGES = True
imgs = os.listdir(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_222223\ZS')
index = 0
imgs.sort()
print(imgs)

for img in imgs:
    num = img.split('.')[0]

    
    rgb_img2 = Image.open(os.path.join(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_222223\ZS', img))
    # exif_dict = piexif.load(rgb_img2.info["exif"])

    width, height = rgb_img2.size

    crop_width, crop_height = 440, 440  # Adjust as needed

    # Crop the image
    
    cropped_rgb_image2 = rgb_img2.crop((260, 260, rgb_img2.width - 260, rgb_img2.height - 260))
    
    # left, top, right, bottom

    cropped_rgb_image2.save(os.path.join(r'd:\Research\2- Research (DeepForest)\denisty_check_final_revision\Scene_222223\ZS_cropped', img))
    index += 1


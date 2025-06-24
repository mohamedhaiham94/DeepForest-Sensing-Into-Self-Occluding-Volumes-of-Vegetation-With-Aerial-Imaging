import cv2
import numpy as np
from skimage.metrics import mean_squared_error

# Load the two images (both images must be the same size)
image1 = cv2.imread(r'd:\Research\2-Paper\Results_real_data\data\channels_data\Final Results\New folder\center2.png', cv2.IMREAD_GRAYSCALE) / 255
image2 = cv2.imread(r'd:\Research\2-Paper\Results_real_data\data\channels_data\Final Results\New folder\PC.png', cv2.IMREAD_GRAYSCALE) / 255

# Get dimensions of the images
rows1, cols1 = image1.shape
rows2, cols2 = image2.shape

# Initialize variables to track minimum MSE and best parameters
min_mse = float('inf')
best_scale = 1.0
best_tx, best_ty = 0, 0

# Define initial guess for scaling and translation
initial_scale = 1.0
initial_tx = 170
initial_ty = 152

# Define a range for scaling and translations
scales = np.linspace(initial_scale - 0.2, initial_scale + 0.2, 100)  # Test scales around initial guess
translation_range_x = range(initial_tx - 10, initial_tx + 11, 1)  # Test translations around initial guess in X
translation_range_y = range(initial_ty - 10, initial_ty + 11, 1)  # Test translations around initial guess in Y

# Brute force search for the best scaling and translation
for scale in scales:
    # Scale the second image
    # scaled_image2 = cv2.resize(image2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_image2 = image2

    for tx in translation_range_x:
        for ty in translation_range_y:
            # Determine the overlapping region
            x_start1 = max(0, -tx)
            y_start1 = max(0, -ty)
            x_end1 = min(cols1, cols2 - tx)
            y_end1 = min(rows1, rows2 - ty)

            x_start2 = max(0, tx)
            y_start2 = max(0, ty)
            x_end2 = x_start2 + (x_end1 - x_start1)
            y_end2 = y_start2 + (y_end1 - y_start1)

            if x_end1 <= x_start1 or y_end1 <= y_start1:
                continue

            # Extract the overlapping regions
            region1 = image1[y_start1:y_end1, x_start1:x_end1]
            region2 = scaled_image2[y_start2:y_end2, x_start2:x_end2]

            # Create a mask to exclude pixels with value 255 (if any)
            mask = (region1 != 255) & (region2 != 255)

            # Apply the mask to both regions
            masked_region1 = region1[mask]
            masked_region2 = region2[mask]

            # Skip computation if no valid pixels remain
            if len(masked_region1) == 0 or len(masked_region2) == 0:
                continue

            # Compute MSE
            mse = mean_squared_error(masked_region1, masked_region2)

            # Update the minimum MSE and parameters if needed
            if mse < min_mse:
                min_mse = mse
                best_scale = scale
                best_tx, best_ty = tx, ty

# Apply the optimal transformation to the second image
scaled_image2 = cv2.resize(image1, None, fx=best_scale, fy=best_scale, interpolation=cv2.INTER_LINEAR)
translation_matrix = np.float32([[1, 0, best_tx], [0, 1, best_ty]])
aligned_image2 = cv2.warpAffine(scaled_image2, translation_matrix, (cols1, rows1))

# Save the transformed image
cv2.imwrite('aligned_image2.png', aligned_image2 * 255)

# Output the results
print(f"Minimum MSE: {min_mse}")
print(f"Best Scale: {best_scale}, Best Translation X: {best_tx}, Best Translation Y: {best_ty}")
print("The aligned image is saved as 'aligned_image2.png'.")

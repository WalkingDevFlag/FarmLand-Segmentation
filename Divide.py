import os
import numpy as np
from PIL import Image

# Set the input image path
image_path = "E:\\Random Python Scripts\\Debris-Segmentation\\Dataset\\Video Data\\Video 11\\frame_0007.jpg"
# Set the output folder path
output_folder = "E:\\Random Python Scripts\\Debris-Segmentation\\Dataset\\New Dataset 44"

# Open the image
img = Image.open(image_path)

# Calculate the number of tiles
num_tiles_x = int(np.ceil(img.width / 512))
num_tiles_y = int(np.ceil(img.height / 512))

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each tile
for i in range(num_tiles_x):
    for j in range(num_tiles_y):
        # Calculate the tile coordinates
        x1 = i * 512
        y1 = j * 512
        x2 = min(x1 + 512, img.width)
        y2 = min(y1 + 512, img.height)

        # Crop the tile
        tile = img.crop((x1, y1, x2, y2))

        # Save the tile
        tile.save(os.path.join(output_folder, f'tile_{i}_{j}.png'))

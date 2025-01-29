#########

# This file iterates over a tif volume and breaks each img into 12 chunks
# Saves chunks into a directory for training

#########

from PIL import Image
import tifffile as tiff
import numpy as np
import os

print(os.getcwd())
tiff_path = '/home/hice1/dharden7/scratch/sd_proj/sd/Stable_diff/volumedata.tif'
output_dir = '/home/hice1/dharden7/scratch/sd_proj/sd/Stable_diff/sd_models/complete_tdata'
os.makedirs(output_dir, exist_ok=True)

with tiff.TiffFile(tiff_path) as tif:
    for i, page in enumerate(tif.pages):
        img = page.asarray()
      
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                img = np.mean(img[:, :, :3], axis=2)  
            elif img.shape[2] == 3:  # RGB
                img = np.mean(img, axis=2)
        # Ensure the image is in uint8 format
        img = img.astype(np.uint8)

        # Get image dimensions
        img_height, img_width = img.shape
        
        tile_size = 256

        num_tiles_x = img_width // tile_size  # Should be 4
        num_tiles_y = img_height // tile_size  # Should be 3

        tile_count = 0  
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                start_x = x * tile_size
                start_y = y * tile_size
                end_x = start_x + tile_size
                end_y = start_y + tile_size

                tile = img[start_y:end_y, start_x:end_x]

                img_tile = Image.fromarray(tile)

                output_path = os.path.join(output_dir, f'image_{i:03}_tile_{tile_count:02}.png')
                img_tile.save(output_path, format='PNG')

                print(f'Saved {output_path}')
                tile_count += 1

print("All images processed and saved.")

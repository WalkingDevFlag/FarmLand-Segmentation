import os
import cv2
import numpy as np

# Function to parse polygon coordinates from label file
def parse_label_file(label_path, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = line.strip().split()
            points = []
            for i in range(1, len(coords), 2):  # Start from index 1 to skip the class identifier
                x = float(coords[i])
                y = float(coords[i + 1])
                points.append((int(x * image_shape[1]), int(y * image_shape[0])))  # Scale coordinates to image size

            points = np.array(points, dtype=np.int32)
            if len(points) > 2:  # Ensure there are enough points to form a polygon
                cv2.fillPoly(mask, [points], color=(255, 255, 255))  # Fill polygon with white (255)

    return mask

# Main function to generate masks for all images
def generate_masks(images_dir, labels_dir, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)
    image_files = os.listdir(images_dir)

    for image_file in image_files:
        if image_file.endswith('.jpg'):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))

            if os.path.exists(label_file):
                image = cv2.imread(image_path)
                mask = parse_label_file(label_file, image.shape[:2])  # Use only width and height for image shape

                # Ensure mask is not empty (not all black)
                if np.max(mask) > 0:
                    mask_filename = os.path.join(masks_dir, image_file.replace('.jpg', '_mask.jpg'))
                    cv2.imwrite(mask_filename, mask)
                    print(f"Mask generated and saved for {image_file}")
                else:
                    print(f"Skipping {image_file} due to empty mask")

# Paths configuration (adjust these paths according to your folder structure)
images_folder = r'E:\Sid Folder\Random Python Scripts\Farm Land Detection\farm_data\images'
labels_folder = r'E:\Sid Folder\Random Python Scripts\Farm Land Detection\farm_data\labels'
masks_folder = r'E:\Sid Folder\Random Python Scripts\Farm Land Detection\farm_data\masks'

# Generate masks
generate_masks(images_folder, labels_folder, masks_folder)

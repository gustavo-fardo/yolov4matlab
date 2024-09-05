import os
import scipy.io
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset directory for YOLO labels and images.')
parser.add_argument('dataset_dir', type=str, help='Path to the dataset directory')

# Parse command line arguments
args = parser.parse_args()
dataset_dir = args.dataset_dir

# Define paths based on the dataset directory
image_dir = os.path.join(dataset_dir, 'train', 'images')
label_dir = os.path.join(dataset_dir, 'train', 'labels')
output_mat_file = os.path.join(dataset_dir, 'rede_copel.mat')

# Prepare a list to store structured data
data_list = []

# Loop through each label file
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        # Corresponding image file path
        image_name = label_file.replace('.txt', '.jpg')  # Assuming .jpg format
        image_path = os.path.join(image_dir, image_name)
        
        # Read the label file
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f.readlines():
                line_data = line.strip().split()
                class_id = int(line_data[0])
                center_x = float(line_data[1])
                center_y = float(line_data[2])
                width = float(line_data[3])
                height = float(line_data[4])

                # Convert YOLO format (center_x, center_y, width, height) to (xmin, ymin, width, height)
                boxes = [center_x, center_y, width, height]
                # Structure the data for this image
                data_list.append([image_path, boxes, [class_id]])

# Convert the data list to a structured array format suitable for MATLAB
structured_data = {
    'path': [item[0] for item in data_list],
    'bbox': [item[1] for item in data_list],
    'class': [item[2] for item in data_list]
}

print("Created .mat table for dataset located in " + dataset_dir)

# Save the structured data to a .mat file
scipy.io.savemat(output_mat_file, {'rede_copel': structured_data})
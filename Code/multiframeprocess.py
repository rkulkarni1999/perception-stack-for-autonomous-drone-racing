import cv2
import numpy as np
from unet import MakeUnet
from unet import get_prediction_image, imshow_gray
import torch 
from PIL import Image 
import os
from imgprocessing import processimage
import re

# Define the source folder
source_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/submission_frames2'

# Create the output folders if they don't exist
grayscale_output_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/grayscale_predicted_images'
if not os.path.exists(grayscale_output_folder):
    os.makedirs(grayscale_output_folder)

rgb_output_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/orientation_images'
if not os.path.exists(rgb_output_folder):
    os.makedirs(rgb_output_folder)

#######################
# IMPORTANT FUNCTION
#######################
device = 'cuda:0'
checkpoint = torch.load('/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/Unetmodel/my_checkpoint_rutwik_training3.pth.tar')
net, lossfun, optimizer = MakeUnet(False)
model = net
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# List all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.startswith('photo') and f.endswith('.jpg')]
# Sort the image files in order
image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# def sort_key(filename):
#     # Use regular expression to find all numbers in the filename
#     numbers = re.findall(r'\d+', filename)
#     # Convert the first group of digits to an integer
#     return int(numbers[0]) if numbers else 0

# # Sort the list of files using the custom sort key
# image_files = sorted(image_files, key=sort_key)

for index, image_file in enumerate(image_files, start=1):
    # Load the image
    image_path = os.path.join(source_folder, image_file)
    image = Image.open(image_path)
    print(f"IMAGE GOING INSIDE  : {type(image)}")
    cvimage = cv2.imread(image_path)
    cvimage = cv2.resize(cvimage, (480, 360))
    print(f"Processing {image_file}...")

    pred = get_prediction_image(image, model, checkpoint, net,device)
    grayscale_image = np.where(pred == 1, 255, 0).astype(np.uint8)

    # Save the grayscale image in the grayscale output folder
    processed_image_path = os.path.join(grayscale_output_folder, f'processed_{index}.png')
    cv2.imwrite(processed_image_path, grayscale_image)
    corners_add_count = 0
    # Process the grayscale image along with the original image
    output_image,rotation_vector, translation_vector = processimage(grayscale_image, cvimage,corners_add_count = 0)
    # im_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Save the processed RGB image in the RGB output folder
    processed_rgb_image_path = os.path.join(rgb_output_folder, f'processed_rgb_{index}.png')
    
    cv2.imwrite(processed_rgb_image_path, output_image)

print("Processing complete.")
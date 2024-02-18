import cv2
from djitellopy import Tello
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from unet import MakeUnet
from unet import get_prediction_image, imshow_gray
import torch
from PIL import Image
import os
from imgprocessing import processimage
from scipy.spatial.transform import Rotation as R

from multiprocessing import Process

#######################################################
# METHOD THAT WILL OUTPUT CONTINUOUS PROCESSED IMAGES
#######################################################
def capture_continuous_photos(output_directory, frame_read_obj):
    count = 0
    while True:
        frame = frame_read_obj.frame
        if frame is not None:
            filename = os.path.join(output_directory, f'photo_{count}.jpg')
            cv2.imwrite(filename, frame)
            count += 1
            time.sleep(1)  # Adjust the sleep time as needed

# Paths
base_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/'
grayscale_output_folder = os.path.join(base_folder, 'grayscale_predicted_images/')
rgb_output_folder = os.path.join(base_folder, 'orientation_images/')
output_directory = os.path.join(base_folder, 'drone_images/')

# Ensure directories exist
os.makedirs(grayscale_output_folder, exist_ok=True)
os.makedirs(rgb_output_folder, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# U-Net Model
device = 'cuda:0'
checkpoint = torch.load('/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/Unetmodel/my_checkpoint_rutwik_training3.pth.tar')
net, lossfun, optimizer = MakeUnet(False)
print(f"Device Available is : {device}")
model = net
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# Initialize Tello drone
tello = Tello()
tello.connect()
tello.takeoff()

battery_level = tello.get_battery()
print(f"Battery level: {battery_level}%")
desired_altitude = 80
speed = 30  # Example speed
tello.go_xyz_speed(0, 0, desired_altitude, speed)
print("Reached desired height")

tello.streamon()
frame_read = tello.get_frame_read(with_queue=True)

####################################################
# STARTING THE PHOTO CAPTURING PROCESS IN PARALLEL
####################################################
photo_process = Process(target=capture_continuous_photos, args=(output_directory, frame_read))
photo_process.start()

count = 0

try:
    while True:
        # Get a frame from the Tello video stream
        drone_frame = frame_read.frame
        if drone_frame is not None:
            # Process the frame
            # ... (The rest of your image processing and Tello control logic) ...
            # For brevity, the main image processing logic is omitted

            pass  # Replace with your image processing and drone control logic

except KeyboardInterrupt:
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('Keyboard interrupt, stopping...')

    ###########################################
    # TERMINATING THE PHOTO CAPTURING PROCESS
    ###########################################
    photo_process.terminate()
    photo_process.join()  # Wait for the photo_process to terminate


    tello.streamoff()
    tello.land()

finally:
    # This ensures we clean up properly even if other exceptions occur
    if photo_process.is_alive():
        photo_process.terminate()
        photo_process.join()
    tello.end()  # Gracefully shutdown the Tello object
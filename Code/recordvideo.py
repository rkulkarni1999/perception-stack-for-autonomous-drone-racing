import cv2
from djitellopy import Tello
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Initialize Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Set the directory where frames will be saved
output_directory = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/videoframes/trial5'

# Ensure the output directory exists
# tello.set_video_fps(Tello.FPS_30)
os.makedirs(output_directory, exist_ok=True)
obj = tello.get_frame_read(with_queue= True)
img = None
count = 0
# cv2.namedWindow('Image Viewer', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Image Viewer', 360, 480)
print("done")
while True:
    try:
        # Get a frame from the Tello video stream
        img = obj.frame
        if img is not None:
            fra = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Save the frame to the output directory with a timestamp
            frame_filename = os.path.join(output_directory, f'frame_{count}.png')
            count +=1
            cv2.resize(fra, (360,480))
            cv2.imwrite(frame_filename, fra)
            # img.resize((360,480))
            
            print(f'frame_{count}')
            # plt.imshow(img)
            # plt.pause(0.001)
            # cv2.imshow('Image Viewer', img)
            # time.sleep(0.03)

    except KeyboardInterrupt:
        # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
        print('keyboard interrupt')
        tello.streamoff
        tello.emergency()
        tello.emergency()
        tello.end()
        break
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
from readenv import get_windows
from set_height import go_to_height
from multiprocessing import Process
from threading import Thread
import threading


#Paths
global grayscale_output_folder,rgb_output_folder,output_directory,env_path,continous_frames

grayscale_output_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/grayscale_predicted_images/'
if not os.path.exists(grayscale_output_folder):
    os.makedirs(grayscale_output_folder)

rgb_output_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/orientation_images/'
if not os.path.exists(rgb_output_folder):
    os.makedirs(rgb_output_folder)

output_directory = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/drone_images/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

continous_frames = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/frames/continous_frames'
if not os.path.exists(continous_frames):
    os.makedirs(continous_frames, exist_ok=True)

env_path = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/worldmap/environment.txt'


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
tello.streamon()
obj = tello.get_frame_read(with_queue= True)
img = None
count = 0

img = obj.frame
if img is not None:
    for i in range(0,6):
        img = obj.frame
    img = cv2.resize(img, (480,360))
    pil_image = Image.fromarray(img)
    pred = get_prediction_image(pil_image, model, checkpoint, net, device)

print("cuda intialized")

tello.takeoff()
battery_level = tello.get_battery()
print(f"Battery level: {battery_level}%")
print("reched desired height")

def capture_continuous_photos(output_directory, frame_read_obj, stop_event):
    count = 0
    while not stop_event.is_set():
        frame = frame_read_obj.frame
        if frame is not None:
            filename = os.path.join(output_directory, f'photo_{count}.jpg')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, frame)
            count += 1



def get_center(drone_frame,count):
    count +=1
    frame_filename = os.path.join(output_directory, f'frame_{count}.png')
    
    bgr_frame = cv2.cvtColor(drone_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(frame_filename, bgr_frame)
    print(f'saved rgb_frame_{count}')
    
    #corner prediction

    drone_frame = cv2.resize(drone_frame, (480,360))
    pil_image = Image.fromarray(drone_frame)
    # plt.imshow(pil_image)
    # plt.pause(1)

    pred = get_prediction_image(pil_image, model, checkpoint, net, device)
    print("got predicted images")
    grayscale_image = np.where(pred == 1, 255, 0).astype(np.uint8)
    processed_image_path = os.path.join(grayscale_output_folder, f'processed_{count}.png')
    cv2.imwrite(processed_image_path, grayscale_image)
    print(f'saved predicted_grayscale_frame_{count}')

    # Process the grayscale image along with the original image
    corners_add_count = 0
    drone_frame = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
    output_image,rotation_vector, translation_vector = processimage(grayscale_image,drone_frame,corners_add_count = 0)
    processed_rgb_image_path = os.path.join(rgb_output_folder, f'processed_rgb_{count}.png')
    cv2.imwrite(processed_rgb_image_path, output_image)
    rotation_vector = np.squeeze(rotation_vector)
    rotation = R.from_rotvec(rotation_vector)
    euler_angles = rotation.as_euler('zyx', degrees= True)  # 'zyx' is the sequence of rotation; change as needed
    print(f'saved gate orientaion frame_{count}')
    print("euler angles=", euler_angles)
    print("translation vector",translation_vector)

    return translation_vector, rotation_vector,count


desired_altitude = 160

# x_pos_shift = [-140,110,-180]
# x_pos_shift = [-70,110,-120]
# x_pos_shift = [-65, 130, -114]
# x_pos_shift = [-70, 130, -114]
x_pos_shift = get_windows(env_path)

# rotate = [22,0,17]

## uncomment for recoding frames
# stop_photo_thread = threading.Event()
# photo_thread = Thread(target=capture_continuous_photos, args=(continous_frames, obj, stop_photo_thread))
# photo_thread.start()
window = [False, False, False] 

for w_count in range(0,3):
    tello.go_xyz_speed(0,x_pos_shift[w_count],0,100)
    go_to_height(tello, desired_altitude)
    # tello.rotate_counter_clockwise(rotate[w_count])
    time.sleep(0.5)
    while window[w_count] == False:
        try:
            # Get a frame from the Tello video stream
            for i in range(0,3):
                drone_frame = obj.frame
            
            if drone_frame is not None:
                
                # Save the frame to the output directory with a timestamp
                translation_vector, rotation_vector,count = get_center(drone_frame,count)
                
                # print(translation_vector.mean())
                if translation_vector.mean() != 0:
                    print(tello.get_height())
                    print(int(translation_vector[0]*0.1),(int(translation_vector[2]*0.1))+30,(int(translation_vector[1]*0.1)))
                    # tello.go_xyz_speed((int(translation_vector[2]*0.1))+ 20,-(int(translation_vector[0]*0.1)), -int(translation_vector[0]*0.1), 40)
                    tello.go_xyz_speed((int(translation_vector[2]*0.1))+ 30,-(int(translation_vector[0]*0.1)), -40,80)
                    window[w_count] = True
                # tello.rotate_clockwise(rotate[w_count])
                time.sleep(0.5)

        except KeyboardInterrupt:
            # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
            print('keyboard interrupt')
            # stop_photo_thread.set()
            # photo_thread.join()
            tello.streamoff= 0
            tello.emergency()
            tello.emergency()
            tello.end()
            tello.land()
            break





    

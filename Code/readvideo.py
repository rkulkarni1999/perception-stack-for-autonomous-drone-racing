import cv2
import os
import time

# Path to the folder containing images
image_folder = '/home/pear/AerialRobotics/Aerial/Unet_sim2real/videoframes/trial2'

# Get a list of image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if not image_files:
    print("No image files found in the specified folder.")
    exit()

# Sort the list of image files
image_files.sort()

# Initialize an index to keep track of the currently displayed image
current_image_index = 0

# Create a window for displaying images
cv2.namedWindow('Image Viewer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image Viewer', 480, 360)

# Set the frame rate (images per second)
frame_rate = 30  # Adjust to your desired frame rate (e.g., 2 images per second)

while True:
    # Read the current image
    image_path = os.path.join(image_folder, image_files[current_image_index])
    image = cv2.imread(image_path)

    if image is not None:
        # Display the image
        cv2.imshow('Image Viewer', image)
        
        # Wait for the specified time to change the image
        time.sleep(1 / frame_rate)

        # Update the index to show the next image
        current_image_index = (current_image_index + 1) % len(image_files)
    else:
        print(f"Error reading image: {image_path}")
        current_image_index = (current_image_index + 1) % len(image_files)

    # Check for a user's request to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
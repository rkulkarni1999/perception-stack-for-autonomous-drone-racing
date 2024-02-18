from djitellopy import tello
import cv2
import matplotlib.pyplot as plt

drone = tello.Tello()

drone.connect()

print(drone.get_battery())

drone.streamon()
fig, ax = plt.subplots()
count = 0
num = 1
obj  = drone.get_frame_read(with_queue=False)
while True:
    print("Getting Frame...")
    img = obj.frame
    print("GOT Frame...")
    if img is not None:
        plt.imshow(img)
        if plt.waitforbuttonpress(0.01):
            break
        fra = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (num%1 ==0):
            cv2.imwrite('/home/pear/AerialRobotics/Aerial/HW3a/src/CameraCalibration-main/ori_images/img' + str(num) + '.png', fra)
            print("image saved!")
        num += 1
    else:
        print("Image not loaded properly.")

    
    # ax.imshow(img)
    # # # Draw the updated frame
    # plt.draw()
    # # plt.pause(0.01)
    # # Check for user input to exit the loop
    #     break
    # img2 = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
    # if count > 10:
    #     cv2.imshow("image",img2)
    #     cv2.waitKey(1)
    # count +=1
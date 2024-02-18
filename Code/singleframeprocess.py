import cv2
import numpy as np
from unet import MakeUnet
from unet import get_prediction_image, imshow_gray
import torch 
from PIL import Image 
import matplotlib.pyplot as plt
from imgprocessing import processimage


#######################
# IMPORTANT FUNCTION
#######################
device = 'cuda:0'
checkpoint = torch.load('/home/pear/AerialRobotics/Aerial/Unet_sim2real/src/Unetmodel/my_checkpoint_rutwik_training3.pth.tar')
net, lossfun, optimizer = MakeUnet(False)
model = net
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
image = Image.open('src/frames/submission_frames2/photo_647.jpg')
cvimage = cv2.imread('src/frames/submission_frames2/photo_647.jpg')
cvimage = cv2.resize(cvimage,(480,360))
print("Got Image !!")

pred = get_prediction_image(image, model, checkpoint, net,device)
grayscale_image = np.where(pred == 1, 255, 0).astype(np.uint8)
plt.imshow(grayscale_image, cmap='gray')
plt.show()
corners_add_count = 0
output_image,rotation_vector,translation_vector = processimage(grayscale_image,cvimage,corners_add_count)
im_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

plt.imshow(im_rgb)
plt.show()

# cv2.imshow('Img', output_image)
# cv2.waitKey(0)


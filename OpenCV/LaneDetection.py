import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



img_dir = f"C:\\Vishram\\Python\\AI\self-driving-car\\project_1_lane_finding_basic\\data\\test_images\\"

for i in os.listdir(img_dir):
    print(img_dir+i)
    #img = Image.open(img_dir+i)
    img = cv2.imread(img_dir+i)
   # resize to 960 x 540
    color_image = cv2.resize(img, (960, 540))

    # convert to grayscale
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # perform gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # perform edge detection
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=150)
    minLineLength = 20
    maxLineGap = 10
    detected_lines = cv2.HoughLinesP(img_edge,1,np.pi/180,100,minLineLength,maxLineGap)
    
   

    plt.imshow(img_edge)
    plt.show()



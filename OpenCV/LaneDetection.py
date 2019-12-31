import cv2
import numpy as np
import matplotlib.pyplot as plt

import os



def Canny(img):
   lane_img = np.copy(img)
   gray_img = cv2.cvtColor(lane_img,cv2.COLOR_RGB2GRAY)
   blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
   canny_img = cv2.Canny(blur_img,150,200,apertureSize = 3)
   return canny_img    


def region_of_intrest(img):
   height = img.shape[0]
   width = img.shape[1]
   print(height)
   polygon = np.array([[(200, height), (width, height), (int((width-200)/2),250)]])
  # r = cv2.selectROI(img);
   # Crop image
  # imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
   
   mask = np.zeros_like(img)
   cv2.fillPoly(mask,polygon,255)
   masked_img = cv2.bitwise_and(img,mask)
   return masked_img


def display_lanes(img, lines):
   line_img = np.zeros_like(img)
   if lines is not None:
          for line in lines:
               x1, y1, x2, y2 = line.reshape(4)
               pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
               cv2.polylines(img, [pts], True, (0,255,0))

   return line_img
       
       
def average_slope_intercept(img, lines):
   left_fit = []
   right_fit = []
   for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      parameters = np.polyfit((x1, x2), (y1, y2), 1)
      slope = parameters[0]
      intercept = parameters[1]
      if slope < 0:
         left_fit.append((slope, intercept))
      else:
         right_fit.append((slope, intercept))
      
   left_fit_average = np.average(left_fit, axis=0)
   right_fit_average = np.average(right_fit, axis=0)
   left_lane = makeCordinate(img, left_fit_average)
   right_lane = makeCordinate(img, right_fit_average)
   return np.array([left_lane, right_lane])
      

def makeCordinate(img, line_Parameter):
   try:
      slope, intercept = line_Parameter
      y1 = img.shape[0]
      y2 = int(y1 * (3/5))
      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)
   except TypeError:
      slope, intercept = 0,0
      y1 = img.shape[0]
      y2 = int(y1 * (3/5))
      x1 = int((y1 - 0)/1)
      x2 = int((y2 - 0)/1)
  
   return np.array([x1, y1, x2, y2])
   



def laneDetect(frame):
   canny_img = Canny(frame)
   cropped_img = region_of_intrest(canny_img)
   lines = cv2.HoughLinesP(cropped_img,cv2.HOUGH_PROBABILISTIC,np.pi/180,100,30,minLineLength=30, maxLineGap=5)
   average_lines = average_slope_intercept(frame,lines)
   lines_img = display_lanes(frame,average_lines)
   planeImg = np.zeros_like(frame)
   combo_img = cv2.addWeighted(frame, 0.8, lines_img, 1, 1)
   return combo_img



def detectVideo():
   cap = cv2.VideoCapture("test2.mp4")
   while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      img = np.copy(frame)
      laneFrame = laneDetect(img)
      # Display the resulting frame
      cv2.imshow('video',laneFrame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break


   # When everything done, release the capture
   cap.release()
   cv2.destroyAllWindows()





detectVideo()
'''
src_img = cv2.imread("./data/2011_09_26/2011_09_26_drive_0001_extract/image_03/data/0000000000.png")
dir = "./data/2011_09_26/2011_09_26_drive_0001_extract/image_03/data/"

for imgFile in os.listdir(dir):
   img = cv2.imread(dir + imgFile)       
   laneFrame = laneDetect(img)    
   cv2.imshow('video',laneFrame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break  
'''
#plt.imshow(Canny(src_img))
#plt.imshow(laneDetect(src_img))
#plt.show()
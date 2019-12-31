import numpy as np
import cv2
import matplotlib.pyplot as plt



img = cv2.imread("Road_in_Norway.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
img = cv2.resize(img,(200,200))
img2 = np.zeros((20,20),dtype=int)
img = np.dot(img,img2)


plt.imshow(img)

plt.show()

print(img.shape)
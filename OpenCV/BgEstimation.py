import numpy as np
import cv2

from skimage import data, filters


cap = cv2.VideoCapture("Traffic.mp4")

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

frames =[]

for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)


cap.set(cv2.CAP_PROP_POS_FRAMES,0)

grayMedianFrame = cv2.cvtColor(medianFrame,cv2.COLOR_BGR2GRAY)

ret = True

while(ret):
    ret, frame =  cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(frame, grayMedianFrame)

    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('binary', dframe)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

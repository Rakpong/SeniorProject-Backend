import numpy as np
import os
import cv2

source = "F:/workkk/SeniorProject-Backend/function/Fire video/"
video = os.listdir(source)
print(source)
cap = cv2.VideoCapture(source+video[2])
total_frames = cap.get(7)
print(total_frames)
gap_time = total_frames/14
print(gap_time)
success, image = cap.read()
count = 0
target_folder = "F:/workkk/SeniorProject-Backend/function/frame/"
os.chdir(target_folder)
time = []
while success:
    if(count > 0):
        cap.set(cv2.CAP_PROP_POS_MSEC, (count*gap_time*35))
        cv2.imwrite("frame%d.jpg" % count, image)
        time.append( (count*gap_time*35) * 0.001)
        # print((count*gap_time*35) * 0.001)
        success, image = cap.read()
        print('Read a new frame: ', success)
    count += 1
os.chdir("F:/workkk/SeniorProject-Backend/function/")
print(time)
# os.remove(source+video[0])
# print("Video Removed!")
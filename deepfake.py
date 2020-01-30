import numpy as np
import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import cv2 as cv

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

video_path = "train_sample_videos/abarnvbtwb.mp4"
capture_image = cv.VideoCapture(video_path)

ret, frame = capture_image.read()

class FaceDetector():

    def __init__(self,faceCascadePath):
        self.faceCascade=cv.CascadeClassifier(faceCascadePath)


    def detect(self, image, scaleFactor=1.1,
               minNeighbors=5,
               minSize=(30,30)):
        
        #function return rectangle coordinates of faces for given image
        rects=self.faceCascade.detectMultiScale(image,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
        return rects

frontal_cascade_path="opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
fd=FaceDetector(frontal_cascade_path)
eye_cascade_path = "opencv-master\data\haarcascades\haarcascade_eye.xml"
ed = FaceDetector(eye_cascade_path)
profile_cascade_path = "opencv-master\data\haarcascades\haarcascade_profileface.xml"
pd = FaceDetector(profile_cascade_path)
smile_cascade_path = "opencv-master\data\haarcascades\haarcascade_smile.xml"
sd = FaceDetector(smile_cascade_path)

def show_image(image):
    plt.figure(figsize=(18,15))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def detect_face(image, scaleFactor, minNeighbors, minSize):
    # face will detected in gray image
    image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces=fd.detect(image_gray,
                   scaleFactor=scaleFactor,
                   minNeighbors=minNeighbors,
                   minSize=minSize)

    for x, y, w, h in faces:
        #detected faces shown in color image
        cv.rectangle(image,(x,y),(x+w, y+h),(127, 255,0),3)

    eyes=ed.detect(image_gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(int(minSize[0]/2), int(minSize[1]/2)))

    for x, y, w, h in eyes:
        #detected eyes shown in color image
        cv.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 0,255),3)

    profiles=pd.detect(image_gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize)

    for x, y, w, h in profiles:
        #detected profiles shown in color image
        cv.rectangle(image,(x,y),(x+w, y+h),(255, 0,0),3)

    show_image(image)



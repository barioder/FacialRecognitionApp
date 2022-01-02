import cv2
import numpy as np
import face_recognition
import os

# to access the images in the folder
path = 'participants'
images = []
classNames = []

# grab the list of images directory
photos = os.listdir(path)
print(photos)

# to import the images one by one
for photo in photos:
    currPhoto = cv2.imread(f'{path}/{photo}')
    images.append(currPhoto)
    classNames.append(os.path.splitext(photo)[0])
print(classNames)

# a function to encode the know photos
def findEcodings (imgList):
    encodeList = []
    for img in imgList:
        # convert the image to black and white
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get the face encoding
        encodedImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodedImg)
    return encodeList


encodeknownList = findEcodings(images)
print('Encoding Complete')

# initialise the webcam
cap = cv2.VideoCapture(0)

# while loop to capture each frame one by one
while True:
    # read frame from a camera
    success, img = cap.read()
    # Resize the captured to a reasonable size
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.Color_BGR2RGB)
    

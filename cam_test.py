# Standard
import time

# Third-party
import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F

# Local
import CNN
 
# load label binarizer
lb = joblib.load('outputs\\lb.pkl')
model = CNN.Cnn()
model.load_state_dict(torch.load('outputs\\model.pth'))
print(model)
print('Model loaded')


def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224,224))
    return hand

cap = cv2.VideoCapture(0) # Using first camera

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # get the hand area on the video capture screen
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    hand = hand_area(frame)
    image = hand
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0)
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    # press `Esc` to exit
    if cv2.waitKey(27) & 0xFF == 27:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

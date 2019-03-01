import cv2
import random as rng
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

model = load_model("strips.model")

camera = cv2.VideoCapture(0)

while True:
  _, frame = camera.read()
  frameClone = frame.copy()

  frameGrayscale = frame.copy()
  frameGrayscale = cv2.cvtColor(frameGrayscale, cv2.COLOR_BGR2GRAY)
  frameGrayscale = cv2.cvtColor(frameGrayscale, cv2.COLOR_GRAY2BGR)

  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
  hue = [29, 85]
  sat = [28, 90]
  lum = [80, 223]
  frame = cv2.inRange(frame, (hue[0], lum[0], sat[0]),  (hue[1], lum[1], sat[1]))
  frame = cv2.dilate(frame, None, iterations=1)
  frame = cv2.erode(frame, None, iterations=1)

  contours = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  index = 0
  for i in range(len(contours)):
    r = cv2.boundingRect(contours[i])
    if cv2.contourArea(contours[i]) < 100 or r[2] > r[3] or r[2] < 40 or r[3] < 40: # Minimum size
      continue

    newRect = (
      (r[0] + (r[2]/2)) - (r[3]/2), r[1], r[3], r[3]
    )

    crop_img = frameGrayscale[newRect[1]:newRect[1]+newRect[3], newRect[0]:newRect[0]+newRect[2]]
    crop_img = cv2.resize(crop_img, (32, 32))

    roi = np.expand_dims(img_to_array(crop_img), axis=0) / 255.0
    preds = model.predict(roi)
    predicted = preds.argmax(axis=1)
    pct = preds[0][predicted[0]] * 100.0
    if pct > 60:
      colors = [
        (0,255,0),
        (0,0,255),
        (255,0,0)
      ]
      cv2.rectangle(frameClone, newRect, colors[predicted[0]], 5)
      cv2.putText(frameClone, str(pct), (newRect[0] + 20, newRect[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

  cv2.imshow('frame', frameClone)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

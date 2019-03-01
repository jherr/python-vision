import cv2
import random as rng
import time

camera = cv2.VideoCapture(0)
sample = 0
prefix = str(time.time())

_, frame = camera.read()
frameClone = frame.copy()

frameGrayscale = frame.copy()
frameGrayscale = cv2.cvtColor(frameGrayscale, cv2.COLOR_BGR2GRAY)

cv2.imshow('grayscale', frameGrayscale)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
hue = [29, 85]
sat = [28, 90]
lum = [100, 223]
frame = cv2.inRange(frame, (hue[0], lum[0], sat[0]),  (hue[1], lum[1], sat[1]))

frame = cv2.dilate(frame, None, iterations=1)
frame = cv2.erode(frame, None, iterations=1)
cv2.imshow('hls', frame)

contours, hierarchy = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

index = 0
for i in range(len(contours)):
  r = cv2.boundingRect(contours[i])
  if cv2.contourArea(contours[i]) < 100 or r[2] > r[3] or r[2] < 40 or r[3] < 40: # Minimum size
    continue

  newRect = (
    (r[0] + (r[2]/2)) - (r[3]/2), r[1], r[3], r[3]
  )

  color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
  cv2.rectangle(frameClone, newRect, color)

  try:
    crop_img = frameGrayscale[newRect[1]:newRect[1]+newRect[3], newRect[0]:newRect[0]+newRect[2]]
    crop_img = cv2.resize(crop_img, (32, 32))
    cv2.imshow('countour {0}'.format(index), crop_img)

    index = index + 1

    cv2.imwrite("samples/" + prefix + "-" + str(sample) + ".jpg", crop_img)
    sample = sample + 1

  except:
    print("An exception occurred")

cv2.imshow('frame', frameClone)

while True:
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

from imutils import paths
import cv2
import os

imagePaths = list(paths.list_images("samples"))

for (i, imagePath) in enumerate(imagePaths):
  name = os.path.basename(imagePath)
  image = cv2.imread(imagePath)
  cv2.imshow("ROI", image)

  key = cv2.waitKey(0)
  if key == ord("`"):
    print("[INFO] ignoring character")
    continue
  if key == ord("q"):
    break
  key = chr(key).upper()

  dirPath = os.path.sep.join(["annotated", key])

  # if the output directory does not exist, create it
  if not os.path.exists(dirPath):
    os.makedirs(dirPath)
  
  cv2.imwrite(dirPath + '/' + name, image)

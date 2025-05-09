import cv2
from cap_dehaze import cap_dehaze

img = cv2.imread("query_hazy_images/outdoor_synthetic/soh5.jpg")
dehazed = cap_dehaze(img)
cv2.imwrite("dehazed_cap.jpg", dehazed)

import cv2
import numpy as np
import matplotlib.pyplot as plt

dark_horse = cv2.imread('../assets/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('../assets/rainbow.jpg')
show_rainbow =cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('../assets/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

# OpenCV BGR ordering
# hist_bricks_blue = cv2.calcHist([blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
# hist_bricks_red = cv2.calcHist([blue_bricks], channels=[2], mask=None, histSize=[256], ranges=[0, 256])

# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#   hist = cv2.calcHist([rainbow], [i], None, [256], [0, 256])
#   plt.plot(hist, color=col)
#   plt.xlim([0, 256])
# plt.title('histogram')

# applying a mask to a color histogram
img = rainbow
mask = np.zeros(img.shape[:2], np.uint8)
mask[600:1000, 600:1000] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)
hist_mask_values_red = cv2.calcHist([rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])

# histogram equalization (higher contrast)
gorilla = cv2.imread('../assets/gorilla.jpg', 0)
hist_values = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
eq_gorilla = cv2.equalizeHist(gorilla)
hist_values = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

gorilla_bgr = cv2.imread('../assets/gorilla.jpg')
# convert image to HSV colorspace to equalize
hsv = cv2.cvtColor(gorilla_bgr, cv2.COLOR_BGR2HSV)

hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
eq_gorilla_w_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


plt.plot(eq_gorilla_w_color)
plt.show()
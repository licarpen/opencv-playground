import cv2
import matplotlib.pyplot as plt

# read in image in gray scale
img = cv2.imread('../assets/rainbow.jpg', 0)

# convert to binary with threshold at middle of color scale
# converts pixels > 127 to white and <=127 to black (use THRESH_BINARY_INV for opposite)
ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# convert all values > threshold to threshold and leave others
# ret, thresh_img2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

# convert all values < threshold to 0 and leave others
# ret, thresh_img3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

plt.imshow(thresh_img, cmap='gray')
plt.show()
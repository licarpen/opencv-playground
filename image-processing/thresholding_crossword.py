import cv2
import matplotlib.pyplot as plt

# read in image in gray scale
img = cv2.imread('../assets/crossword.jpg', 0)

# simple binary threshold
ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# adaptive threshold using pixel neighborhood (block size as odd int) and constant
thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

blended = cv2.addWeighted(src1=thresh_img, alpha=0.6, src2=thresh_img2, beta=0.4, gamma=0)

plt.imshow(blended, cmap='gray')
plt.show()
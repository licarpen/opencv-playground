import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):
  plt.imshow(img, cmap='gray')
  plt.show()

img = cv2.imread('../assets/sudoku.jpg', 0)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# laplacian = cv2.Laplacian(img, cv2.CV_64F)

blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)
display_img(blended)
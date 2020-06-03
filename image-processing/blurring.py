# blurring can be used to improve edge detection by reducing the number of edges detected
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
  img = cv2.imread('../assets/bricks2.jpg').astype(np.float32) / 255
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def display_img(img):
  plt.imshow(img)
  plt.show()

# gamma correction (adjust perceied brightness) < 1: brighter > 1: less bright
i = load_img()
gamma = 3
result = np.power(i, gamma)
# display_img(result)

# blurring: low-ass filter with 2D convolution
# add text to image to help see differentiated b/w techniques

img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)

# value of pixel will be average of all surrounding pixels
kernel = np.ones(shape=(50, 50), dtype=np.float32) / (50 * 50)
dst = cv2.filter2D(img, -1, kernel)
# display_img(dst)

# apply default blurring kernel
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
blurred = cv2.blur(img, ksize=(20, 20))
# display_img(blurred)

# gaussian blurring
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
blurred_gaussian = cv2.GaussianBlur(img, (15, 15), 10)
# display_img(blurred_gaussian)

# median blurring
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255,0,0), thickness=4)
blurred_median = cv2.medianBlur(img, 5)
display_img(blurred_median)

# bilateral filter can be used to reduce noise in image while keeping edges sharp.  SLOW.  

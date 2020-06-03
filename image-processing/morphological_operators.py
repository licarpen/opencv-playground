import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
  blank_img = np.zeros((600, 600))
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(blank_img, text='ABCDE', org=(50, 300), fontFace=font, fontScale=5, color=(255, 255, 255), thickness=25, lineType=cv2.LINE_AA)
  return blank_img

def display_img(img):
  plt.imshow(img, cmap='gray')
  plt.show()


# erosion
img = load_img()
kernel = np.ones((5, 5), dtype=np.uint8)
result = cv2.erode(img, kernel, iterations=3)
# display_img(result)

# opening (removes background noise with erosion followed by dilation)
img = load_img()
white_noise = np.random.randint(low=0, high=2, size=(600, 600)) * 255
noise_image = white_noise + img
kernel = np.ones((5, 5), dtype=np.uint8)
opening = cv2.morphologyEx(noise_image, cv2.MORPH_OPEN, kernel)
#result = cv2.erode(img, kernel, iterations=3)
# display_img(opening)

# for closing (removes foreground noise)use MORPH_CLOSE

# both erosion and dilation
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)
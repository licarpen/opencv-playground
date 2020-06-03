# Generate a mask that is smaller than the original image and overlay

import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('../assets/arete.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../assets/paw.jpg')
img2 = cv2. cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (300, 300))

# upper left corner of mask
x_offset = 3500
y_offset = 2500

rows, columns, channels = img2.shape

# region of interest in original image that will have the added mask
roi = img1[y_offset:y_offset + 300, x_offset:x_offset + 300]

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# build out mask so masked content is pure white (must invert paw image)
mask_inv = cv2.bitwise_not(img2gray)
# add back color channels
white_background = np.full(img2.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
# bring back color
fg = cv2.bitwise_or(img2, img2, mask=mask_inv)

# mask is now content + black background
# blend roi with mask
final_roi = cv2.bitwise_or(roi, fg)

large_img = img1
small_img = final_roi

large_img[y_offset:y_offset + 300, x_offset:x_offset + 300] = small_img

plt.imshow(large_img)
plt.show()



from gettext import install

import matplotlib
import numpy
import pip

pip install opencv-python
pip install numpy
pip install matplotlib
# test image
image = cv2.imread('cat.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram = cv2.calcHist([gray_image], [0],
                         None, [256], [0, 256])

# data1 image
image = cv2.imread('cat.jpeg')
gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram1 = cv2.calcHist([gray_image1], [0],
                          None, [256], [0, 256])

# data2 image
image = cv2.imread('food.jpeg')
gray_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram2 = cv2.calcHist([gray_image2], [0],
                          None, [256], [0, 256])

c1, c2 = 0, 0

# Euclidean Distace between data1 and test
i = 0
while i < len(histogram) and i < len(histogram1):
    c1 += (histogram[i] - histogram1[i]) ** 2
    i += 1
c1 = c1 ** (1 / 2)

# Euclidean Distace between data2 and test
i = 0
while i < len(histogram) and i < len(histogram2):
    c2 += (histogram[i] - histogram2[i]) ** 2
    i += 1
c2 = c2 ** (1 / 2)

if (c1 < c2):
    print("data1.jpg is more similar to test.jpg as compare to data2.jpg")
else:
    print("data2.jpg is more similar to test.jpg as compare to data1.jpg")
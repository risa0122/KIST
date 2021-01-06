import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
img = imread('att_face/s1/1.pgm')
img = img.astype(np.uint8)
img = img / 255
plt.imshow(img,cmap='gray')
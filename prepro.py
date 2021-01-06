
from skimage import io, exposure, restoration, filters, morphology, util
import matplotlib.pyplot as plt
import numpy as np
import nd2reader
import cv2
from scipy.signal import convolve2d as conv2
import numpy as np

nd2 = nd2reader.Nd2("/Users/taco-lab/Desktop/Jongeun/DATA/nd2/JKR12.nd2")
nd2_np = np.array(nd2, dtype=np.uint16)  # (1797,304,304)


vv = []
for i in range(1797):

    a = nd2_np[i]
    y = np.copy(a)
    y = filters.rank.equalize(y, selem=morphology.disk(70))
    y = exposure.equalize_adapthist(y, clip_limit=7)
    y = exposure.equalize_adapthist(y, clip_limit=300)
    y = exposure.equalize_adapthist(y, clip_limit=400)
    y = exposure.equalize_adapthist(y, clip_limit=300)
    y = restoration.denoise_wavelet(y, method='BayesShrink', mode='soft')
    y = np.copy(y) ** 2 * 255
    y = exposure.rescale_intensity(y,out_range=(0, 255))

    markers = np.copy(y)
    markers[y < 70] = 0.000000000000000001


    b = np.copy(markers)
    b[:,:10] = 0.00001
    b[:, 290:] = 0.00001
    b[:42, 10:290] = 0.00001
    b[250:, 10:290] = 0.00001

    #y = util.invert(y)
    y = filters.median(b)
    #y = exposure.adjust_gamma(b, gamma=1.2, gain=150)
    y = exposure.rescale_intensity(y, out_range=(0, 255))


    vv.append(y)
vv = np.array(vv)


for i in range(1797):
    fnames_test = vv[i] # float64
    cv2.imwrite('/Users/taco-lab/PycharmProjects/caiman2/20/{}.tiff'.format(i), fnames_test)

    '''
    hist, hist_centers = exposure.histogram(b)
    # hist1, hist_centers1 = exposure.histogram(y)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(b, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_title('histogram of CLAHE')
'''

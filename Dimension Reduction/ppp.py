from skimage import io, exposure, filters, util, morphology, measure, feature
import nd2reader
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.decomposition import PCA



#nd2 = nd2reader.Nd2("JKR12_mCherry.nd2")
nd2 = nd2reader.Nd2("/Users/taco-lab/Desktop/Jongeun/DATA/nd2/JKR12_DAPI.nd2")

nd2_num = np.array(nd2)
imgs = nd2_num.transpose(1, 2, 0)
x = imgs[:,:,0]
####### local&adapt&medi######
y = filters.rank.equalize(x, selem= morphology.disk(30))
y = exposure.equalize_adapthist(y,clip_limit=10)
y = filters.median(y)
##### regional #####
seed = np.copy(y)
seed[1:-1,1:-1] = seed.min()
masks = y
dilated = morphology.reconstruction(seed,masks,method='dilation')
regi_maxi = y - dilated                                                                                                                                                                                                                                                                                                                                                                                                  - dilated

########## PCA ############
print(regi_maxi.shape)
X = regi_maxi[:,]
pcal = PCA(n_components=1)
X_low = pcal.fit_transform(X)



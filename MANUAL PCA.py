import nd2reader
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io, exposure, filters, util, morphology, measure, feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# nd2 = nd2reader.Nd2("JKR12_mCherry.nd2")
nd2 = nd2reader.Nd2("JKR12_DAPI.nd2")

nd2_num = np.array(nd2)
imgs = nd2_num.transpose(1, 2, 0)
x = imgs[:, :, 0]
####### local&adapt&medi######
y = filters.rank.equalize(x, selem=morphology.disk(30))
y = exposure.equalize_adapthist(y, clip_limit=10)
y = filters.median(y)
##### regional #####
seed = np.copy(y)
seed[1:-1, 1:-1] = seed.min()
masks = y
dilated = morphology.reconstruction(seed, masks, method='dilation')
regi_maxi = y - dilated
X = x
# print(X.max())


##############################################

#### 1. 데이터 정규화 ####

#### 2. 공분산 계산   ####
X = StandardScaler().fit_transform(X)
# print(X.max())
# print(X.shape)

# 또는 한 줄의 numpy
cov_mat = np.cov(X.T)
# print(cov_mat.shape)

#### 3. 고유값 & 벡터 계산 ####

# numpy를 사용하여 고유 값과 벡터를 계산
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print(np.cumsum([i*(100/sum(eig_vals)) for i in eig_vals]))

#### 4. 차원 축소 ####
pc = eig_vecs[0:2]
data_transformed = np.dot(regi_maxi, pc.T)

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(data_transformed.T[0],data_transformed.T[1])
for l,c in zip((np.unique(target)),['red','green','blue']):
    ax0.scatter(data_transformed.T[0,target==l],data_transformed.T[1,target==l],c=c,label=l)
ax0.legend()
plt.show()
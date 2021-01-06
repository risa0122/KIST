import nd2reader
import numpy as np
from skimage import io


dapi = nd2reader.Nd2('JKR12_DAPI.nd2')
cherry = nd2reader.Nd2('JKR12_mCherry.nd2')

dapi_num = np.array(dapi)
cherry_num = np.array(cherry)

print(dapi_num)
print(cherry_num)

plus = (dapi_num + cherry_num) / 2
print(plus.shape)

#io.imshow(plus)
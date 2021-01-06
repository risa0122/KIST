
# https://github.com/flatironinstitute/CaImAn/blob/468428fa9aab257cb6361bf8ad422467b58a0e68/caiman/source_extraction/cnmf/cnmf_optional_outputs.py
T, d1, d2 = images.shape
dims = (d1, d2)

from caiman.source_extraction.cnmf.map_reduce import run_CNMF_patches

A, C, YrA, b, f, sn, optional_outputs = run_CNMF_patches(images.filename, (d1, d2, T), opts)


# https://github.com/flatironinstitute/CaImAn/blob/master/caiman/utils/visualization.py#L321

from caiman.utils.visualization import get_contours

a = get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False)    #(319,)

b = a[0]['coordinates']

b = np.around(b)

gg = np.zeros((304,304))

import pandas as pd

df = pd.DataFrame(b)

df = df.fillna(0)    # 나중에 (0,0) = 0 처리

b = np.array(df)    # np.ndarray (11,2)

bb = [b]    # list (11,2)

# or

bb = []
for i in range(11):
    bb.append((b[i][0],b[i][1]))


for i in range(11):
    gg[int(bb[i][0]),int(bb[i][1])] = 1

gg[0,0] = 0

from skimage import io
io.imshow(gg)






io.imshow(np.reshape(to,(304,304)))


import logging
import matplotlib.pyplot as plt
import numpy as np

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import glob
from imageio import imread
import nd2reader
import cv2

# %%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)


# filename="/tmp/caiman.log"

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

# %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=24, # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)

    # %% First setup some parameters for motion correction
    # dataset dependent parameters

    nd2 = nd2reader.Nd2("/Users/taco-lab/Desktop/Jongeun/DATA/nd2/JKR12.nd2")
    nd2_np = np.array(nd2)  # (1797,304,304)

    # print('fnames[i]',fnames[5].shape)  # (304,304)
    for i in range(1797):
        fnames_test = nd2_np[i]  # float64
        cv2.imwrite('/Users/taco-lab/PycharmProjects/CaImAn/Slicing_nd2/{}.tiff'.format(i), fnames_test)

    fnames = [sorted(glob.glob('/Users/taco-lab/PycharmProjects/CaImAn/Slicing_nd2/*.tiff'))]
    # print('ffnp',ffnp)
    filename_reorder = fnames

    fr = 10  # movie frame rate
    decay_time = 0.4  # length of a typical transient in seconds

    fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                               order='C', border_to_0=0, dview=dview)
    # print("file's name: ",fname_new)

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    print('이미지 형태 : ', images.shape)  # (1797,304,304)
    # print('이미지',images)

    # %% Parameters for source extraction and deconvolution (CNMF-E algorithm)

    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
    Ain = None  # possibility to seed with predetermined binary masks
    merge_thr = .7  # merging threshold, max correlation allowed
    rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20  # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2  # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1  # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0  # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0  # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .8  # min peak value from correlation image
    min_pnr = 10  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background
    ring_size_factor = 1.5  # radius of ring is gSiz*ring_size_factor
    bord_px = 0

    dims = fnames[0, :, :]  # (304,304)
    print(dims)

    opts = params.CNMFParams(params_dict={'dims': dims,
                                          'method_init': 'corr_pnr',  # use this for 1 photon
                                          'K': K,
                                          'gSig': gSig,
                                          'gSiz': gSiz,
                                          'merge_thr': merge_thr,
                                          'p': p,
                                          'tsub': tsub,
                                          'ssub': ssub,
                                          'rf': rf,
                                          'stride': stride_cnmf,
                                          'only_init': True,  # set it to True to run CNMF-E
                                          'nb': gnb,
                                          'nb_patch': nb_patch,
                                          'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
                                          'low_rank_background': low_rank_background,
                                          'update_background_components': True,
                                          # sometimes setting to False improve the results
                                          'min_corr': min_corr,
                                          'min_pnr': min_pnr,
                                          'normalize_init': False,  # just leave as is
                                          'center_psf': True,  # leave as is for 1 photon
                                          'ssub_B': ssub_B,
                                          'ring_size_factor': ring_size_factor,
                                          'del_duplicates': True,  # whether to remove duplicates from initialization
                                          'border_pix': bord_px})

    # %% compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile

    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
    # if your images file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing images[::1] to
    # images[::5] or something similar to compute on a subset of the data

    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    # print('씨앤필터',cn_filter.shape)

    # print parameters set above, modify them if necessary based on summary images
    print(min_corr)  # min correlation of peak (from correlation image)
    print(min_pnr)  # min peak to noise ratio

    # %% RUN CNMF ON PATCHES
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)

    # %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #    cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #    cnm1.fit_file(motion_correct=True)

    # %% DISCARD LOW QUALITY COMPONENTS
    min_SNR = 2.5  # adaptive way to set threshold on the transient size
    r_values_min = 0.85  # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))

    # %% PLOT COMPONENTS
    cnm.dims = dims
    display_images = True  # Set to true to show movies and images
    print('인덱스: ', cnm.estimates.idx_components.dtype)
    if display_images:
        cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
        cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)

    plt.show(cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components))

    # %% MOVIES
    display_images = False  # Set to true to show movies and images
    if display_images:
        # fully reconstructed movie
        cnm.estimates.play_movie(images, q_max=99.5, magnification=2,
                                 include_bck=True, gain_res=10, bpx=bord_px)
        # movie without background
        cnm.estimates.play_movie(images, q_max=99.9, magnification=2,
                                 include_bck=False, gain_res=4, bpx=bord_px)

    # %% STOP SERVER
    cm.stop_server(dview=dview)


# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
# if __name__ == "__main__":
main()

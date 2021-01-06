import os
import glob
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
import scipy


fld = '/Users/taco-lab/PycharmProjects/caiman2/1797'  # path to folder where the data is located
fls = glob.glob(os.path.join(fld,'*.tiff'))  #  change tif to the extension you need
fls.sort()  # make sure your files are sorted alphanumerically

#c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
#    n_processes=24,  # number of process to use, if you go out of memory try to reduce this one
#    single_thread=False)

fr = 10
decay_time = 0.4

motion_correct = False
pw_rigid = False
gSig_filt = (3,3)
border_nan = 'copy'

mc_dict = {
    'fnames': fls,
    'fr' : fr,
    'decay_time': decay_time,
    'motion_correct' : motion_correct,
    'border_nan' :border_nan
}

opts = params.CNMFParams(params_dict=mc_dict)

fname_new = cm.save_memmap([fls], base_name='memmap_', order='C', border_to_0=0)

Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')

K = None
gSig = (5,5)
gSiz = (21,21)
merge_thr = .7
p = 1
stride_cnmf = 20
tsub = 1
ssub = 1
low_rank_background = None
gnb = 0
nb_patch = 0
rf = 40
min_corr = .8      # min peak value from correlation image
min_pnr = 4       # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 2.5
bord_px = 1

p_dict = {
    'dims' : dims,
    'method_init': 'corr_pnr',
    'K' : K,
    'gSig': gSig,
    'gSiz': gSiz,
    'merge_thr':merge_thr,
    'p' : p,
    'tsub' : tsub,
    'ssub' : ssub,
    'rf' : rf,
    'stride' : stride_cnmf,
    'only_init' : True,
    'nb' : gnb,
    'nb_patch' : nb_patch,
    'method_deconvolution' : 'oasis',
    'low_rank_background': low_rank_background,
    'update_background_components': False,
    'min_corr': min_corr,
    'min_pnr': min_pnr,
    'normalize_init': False,               # just leave as is
    'center_psf': True,                    # leave as is for 1 photon
    'ssub_B': ssub_B,
    'ring_size_factor': ring_size_factor,
    'del_duplicates': True,                # whether to remove duplicates from initialization
    'border_pix': bord_px,
    'min_SNR' : 1.0,
    'r_values_min' : 0.1
}

opts.change_params(params_dict=p_dict)

cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
inspect_correlation_pnr(cn_filter, pnr)

Ain=None

cnm = cnmf.CNMF(n_processes=1, params=opts)
cnm.fit(images)

min_SNR = 1.0           # adaptive way to set threshold on the transient size
r_values_min = 0.1    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})

cnm.estimates.evaluate_components(images, cnm.params)



print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

cnm.dims = dims
display_images = True           # Set to true to show movies and images
if display_images:
    cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
    cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)

display_images = True           # Set to true to show movies and images
if display_images:
    # fully reconstructed movie
    cnm.estimates.play_movie(images, q_max=99.5, magnification=2,
                             include_bck=True, gain_res=10, bpx=bord_px)
    # movie without background
    cnm.estimates.play_movie(images, q_max=99.9, magnification=2,
                             include_bck=False, gain_res=4, bpx=bord_px)










import inspect
import numpy
import matplotlib.pyplot as pyplot

import microscPSF.microscPSF as msPSF

# Load and print the default microscope parameters.
for key in sorted(msPSF.m_params):
    print(key, msPSF.m_params[key])
print()

# You can find more information about what these are in this file:
print(inspect.getfile(msPSF))

# We'll use this for drawing PSFs.
#
# Note that we display the sqrt of the PSF.
#
def psfSlicePics(psf, sxy, sz, zvals, pixel_size = 0.05):
    ex = pixel_size * 0.5 * psf.shape[1]

    fig = pyplot.figure(figsize = (12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(numpy.sqrt(psf[sz,:,:]),
               interpolation = 'none',
               extent = [-ex, ex, -ex, ex],
               cmap = "gray")
    ax1.set_title("PSF XY slice")
    ax1.set_xlabel(r'x, $\mu m$')
    ax1.set_ylabel(r'y, $\mu m$')

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(numpy.sqrt(psf[:,:,sxy]),
               interpolation = 'none',
               extent = [-ex, ex, zvals.max(), zvals.min()],
               cmap = "gray")
    ax2.set_title("PSF YZ slice")
    ax2.set_xlabel(r'y, $\mu m$')
    ax2.set_ylabel(r'z, $\mu m$')

    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(numpy.sqrt(psf[:,sxy,:]),
               interpolation = 'none',
               extent = [-ex, ex, zvals.max(), zvals.min()],
               cmap = "gray")
    ax3.set_title("PSF XZ slice")
    ax3.set_xlabel(r'x, $\mu m$')
    ax3.set_ylabel(r'z, $\mu m$')

    pyplot.show()


# Radial PSF
mp = msPSF.m_params
pixel_size = 0.05
rv = numpy.arange(0.0, 3.01, pixel_size)
zv = numpy.arange(-1.5, 1.51, pixel_size)

psf_zr = msPSF.gLZRFocalScan(mp, rv, zv,
                             pz = 0.1,       # Particle 0.1um above the surface.
                             wvl = 0.7,      # Detection wavelength.
                             zd = mp["zd0"]) # Detector exactly at the tube length of the microscope.

fig, ax = pyplot.subplots()

ax.imshow(numpy.sqrt(psf_zr),
          extent=(rv.min(), rv.max(), zv.max(), zv.min()),
          cmap = 'gray')
ax.set_xlabel(r'r, $\mu m$')
ax.set_ylabel(r'z, $\mu m$')

pyplot.show()

# XYZ PSF
psf_xyz = msPSF.gLXYZFocalScan(mp, pixel_size, 31, zv, pz = 0.1)

psfSlicePics(psf_xyz, 15, 30, zv)

# Radial PSF
mp = msPSF.m_params
pixel_size = 0.05
rv = numpy.arange(0.0, 3.01, pixel_size)
pv = numpy.arange(0.0, 3.01, pixel_size) # Particle distance above coverslip in microns.

psf_zr = msPSF.gLZRParticleScan(mp, rv, pv,
                                wvl = 0.7,      # Detection wavelength.
                                zd = mp["zd0"], # Detector exactly at the tube length of the microscope.
                                zv = -1.5)      # Microscope focused 1.5um above the coverslip.

fig, ax = pyplot.subplots()

ax.imshow(numpy.sqrt(psf_zr),
          extent=(rv.min(), rv.max(), pv.max(), pv.min()),
          cmap = 'gray')
ax.set_xlabel(r'r, $\mu m$')
ax.set_ylabel(r'z, $\mu m$')

pyplot.show()

# XYZ PSF
psf_xyz = msPSF.gLXYZParticleScan(mp, pixel_size, 31, pv, zv = -1.5)

psfSlicePics(psf_xyz, 12, 30, pv)
#Script to make channel maps around all identified lines in a given source

import sys
import numpy as np
import glob
import traceback

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import colors
from matplotlib.colors import LogNorm

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from pylab import rcParams
rcParams['figure.dpi'] = 150
rcParams['font.size'] = 6

c = 2.9979e5


def plotChanMaps(infile, sourcename, line, restfreq, maxflux, fluxfactor, vR, vWind, fileout, cmap='rainbow'):
    """
    Plot channel maps
    
    Inputs:
    infile = reduced data cube; format: str ending in .fits
    sourcename = name of source; format: str
    line = line name; format: str
    restfreq = line rest frequency in GHz; format: float
    maxflux = max expected line flux in Jy; format: float
    vWind = source wind velocity in km/s; format: float
    fileout = output file name to save the channel map to; format: str
    
    cmap = color map to use for the channel map plots; format = str
    """

    step = 1    
    
    #Opening input
    with fits.open(infile) as hdul:
        img = hdul[0].data[0, :, :, :]
        hdr = hdul[0].header
        try:
            beams = hdul[1].data
        except:
            pass

    #Spectum scale
    fsize = hdr['NAXIS3']
    fref = hdr['CRVAL3']
    fdelt = hdr['CDELT3']
    fpix = hdr['CRPIX3']
    freq = (np.arange(fsize)-fpix)*fdelt+fref
    freq *= 1e-9

    #Map scale
    npix_x = hdr['NAXIS1']
    npix_y = hdr['NAXIS2']
    pixsize_x = hdr['CDELT1']
    pixsize_y = hdr['CDELT2']
    ra1D = (np.arange(npix_x)-img.shape[2]/2.)*pixsize_x*3600
    dec1D = (np.arange(npix_y)-img.shape[1]/2.)*pixsize_y*3600
    X, Y = np.meshgrid(ra1D, dec1D)

    #Finding range in channel number 
    vel_range = vWind+5
    obsfreq = restfreq/(1+vR/c)

    start_chan = np.argmin(np.abs(freq - obsfreq*(1-vel_range/c)))
    stop_chan = np.argmin(np.abs(freq - obsfreq*(1+vel_range/c)))
    start_chan = np.min([start_chan, freq.size])
    stop_chan = np.max([stop_chan, 0])
    if start_chan > stop_chan:
        a = start_chan
        start_chan = stop_chan
        stop_chan = a

    #min/max
    maxflux = maxflux/5  #**maxflux read in from <source>_lineIDs.csv, in Jy (or something like it), conversion to Jy/beam varies pretty widely :(
    size5asec = int(2.5/(np.mean([abs(pixsize_x), abs(pixsize_y)])*3600))  #npixels in 5"
    max_5asec = 0.0
    #Find maximum flux in central 5"x5"
    for each in img[start_chan:stop_chan, :, :]:
        if np.nanmax(each[int(npix_y/2)-size5asec:int(npix_y/2)+size5asec, int(npix_x/2)-size5asec:int(npix_x/2)+size5asec]) > max_5asec:
            max_5asec = float(np.nanmax(each[int(npix_y/2)-size5asec:int(npix_y/2)+size5asec, int(npix_x/2)-size5asec:int(npix_x/2)+size5asec]))

    vmin = 0.0 #-0.05*maxflux
    vmax = max_5asec*fluxfactor
    size = 10.0

    fig = plt.figure()
    axes = []

    nmaps = int((stop_chan-start_chan)/step)
    ncol = int(np.sqrt(nmaps))
    nrow = int(np.ceil(nmaps/ncol))

    #Beam
    if 'BMAJ' in hdr:
        major = hdr['BMAJ']*3600
        minor = hdr['BMIN']*3600
        pa = hdr['BPA']
    else:
        major = [a[0] for a in beams]
        minor = [a[1] for a in beams]
        pa = [a[2] for a in beams]
        major = np.median(major)
        minor = np.median(minor)
        pa = np.median(pa)

    beam = Ellipse((-size*0.8, -size*0.8), major, minor, angle=90-pa)
    beam.set_facecolor('w')
    beam.set_edgecolor('w')


    n_vel0 = [5,0] #vel closest to vR, n
    for n in range(nmaps):
        #Plotting
        ax = plt.subplot2grid((nrow, ncol), (int(n/ncol), np.mod(n, ncol)), fig=fig)
        cim = ax.imshow(img[n*step+start_chan, :, :], cmap=cmap, origin='lower', \
            vmin=vmin, vmax=vmax, extent=[ra1D.max(), ra1D.min(), dec1D.min(), dec1D.max()], \
            interpolation='spline36')

        #Velocity
        cur_vel = (1-freq[n*step+start_chan]/restfreq)*c
        ax.text(0.95, 0.95, '{:.1f}'.format(cur_vel), color='w', fontsize=4, \
            fontweight='bold', horizontalalignment='right', verticalalignment='top', \
            transform=ax.transAxes, bbox=dict(boxstyle='square,pad=0.3', facecolor='k', alpha=0.6))

        #Finding map closest to restfreq
        if abs(cur_vel-vR) < abs(n_vel0[0]):
            n_vel0[0] = abs(cur_vel-vR)
            n_vel0[1] = n

        #Cosmetics
        ax.set_xlim(size, -size)
        ax.set_ylim(-size, size)
        if np.mod(n, ncol) != 0 or int(n/ncol) != nrow-1:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if np.mod(n, ncol) == 0 and int(n/ncol) == nrow-1:
            ax.set_xlabel('Relative RA (arcsec)')
            ax.set_ylabel('Relative Dec. (arcsec)')
            ax.add_artist(beam)
        if n == nmaps-1:
            bbox_ax_last = ax.get_position()
        ax.tick_params(axis='both', colors='k', labelcolor='k', direction='out', \
            bottom=True, top=True, right=True, left=True)
        axes.append(ax)

    #Set plot of central velocity to have magenta border
    axes[n_vel0[1]].spines['bottom'].set_color('magenta')
    axes[n_vel0[1]].spines['top'].set_color('magenta')
    axes[n_vel0[1]].spines['left'].set_color('magenta')
    axes[n_vel0[1]].spines['right'].set_color('magenta')
    try:
        axes[n_vel0[1]+ncol].spines['top'].set_color('magenta')
        axes[n_vel0[1]+ncol].tick_params(axis='x', which='major', top=False)
    except:
        pass

    p0 = axes[0].get_position().get_points().flatten() #position of top left subplot
    p1 = axes[-1-np.mod(nmaps, ncol)].get_position().get_points().flatten() #position of bottom right subplot
    ax_cbar = fig.add_axes([p0[0]+0.2, 0.022, p1[2]-p0[0]-0.4, 0.02]) #left, bottom, width, height
    fig.colorbar(cim, cax=ax_cbar, orientation='horizontal', label='Flux (Jy/beam)')
    fig.subplots_adjust(hspace=0.0, wspace=-0.52)
    plt.savefig(fileout, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    return


## Plot central features
#plotChanMaps('piGru.12CO.image.pbcor.fits', 'pi1_Gru','CO_3-2', 345.8, 6.3e-2, 1.0, -12, 15, 'pi1_Gru_CO32_chmaps.pdf', cmap='jet')

## Plot fast wind
plotChanMaps('piGru.12CO.image.pbcor.fits', 'pi1_Gru','CO_3-2', 345.8, 6.3e-2, 0.1, -12, 42, 'pi1_Gru_CO32_chmaps_extended.pdf', cmap='jet')


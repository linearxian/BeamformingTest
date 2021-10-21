# -*- coding: utf-8 -*-

import acoular
import matplotlib.pyplot as plt


micgeofile = "array_9.xml"
micgeofile2 = "array_10.xml"
filt_freq = 15000
mic_data1 = "/media/xian/Data/dataset/Bekaert/h5files/44.h5"
mic_data2 = "/media/xian/Data/dataset/Bekaert/h5files/45.h5"
mic_data3 = "/media/xian/Data/dataset/Bekaert/h5files/46.h5"


ts = acoular.MaskedTimeSamples( name=mic_data1 )
ps = acoular.PowerSpectra( time_data=ts, block_size=1024, window="Hanning" )
mg = acoular.MicGeom( from_file=micgeofile2 )
rg = acoular.RectGrid( x_min=-0.2, x_max=4,
                       y_min=-0.2, y_max=4,
                       z=0.5, increment=0.01 )
st = acoular.SteeringVector( grid=rg, mics=mg, steer_type='true location', ref=[1.88,1.87,1.68] )
bb = acoular.BeamformerFunctional( freq_data=ps, steer=st, gamma=50 )
# bb.r_diag = False
Lm = acoular.L_p( bb.synthetic(filt_freq,3) )

# 2 lm
ts = acoular.MaskedTimeSamples( name=mic_data2 )

mg = acoular.MicGeom( from_file=micgeofile2 )
st = acoular.SteeringVector( grid=rg, mics=mg, steer_type='true location', ref=[1.88,1.87,1.68] )

ps = acoular.PowerSpectra( time_data=ts, block_size=1024, window="Hanning" )
bb = acoular.BeamformerFunctional( freq_data=ps, steer=st, gamma=50 )
Lm2 = acoular.L_p( bb.synthetic(filt_freq,3) )

# 3 lm
# ts = acoular.MaskedTimeSamples( name=mic_data2, invalid_channels=[9] )
# ps = acoular.PowerSpectra( time_data=ts, block_size=1024, window="Hanning" )
# bb = acoular.BeamformerFunctional( freq_data=ps, steer=st, gamma=50 )
# Lm3 = acoular.L_p( bb.synthetic(filt_freq,3) )

# results = [Lm, Lm2, Lm3]
results = [Lm, Lm2]

i = 0
fig, axes = plt.subplots(nrows=1, ncols=2)
for res, ax in zip(results, axes.flat):
    im = ax.imshow( res.T, origin="lower", vmin=45, vmax=60, extent=rg.extend() )
    ax.scatter(3.76, 3.735, c='r')
    ax.scatter(3.76, 1.8675, c='r')
    ax.scatter(3.76, 0, c='r')
    ax.scatter(1.88, 3.735, c='r')
    ax.scatter(1.88, 1.8675, c='r')
    ax.scatter(1.88, 0, c='r')
    ax.scatter(0, 3.735, c='r')
    ax.scatter(0, 1.8675, c='r')
    ax.scatter(0, 0, c='r')
    
    ax.scatter(0.38, 1.9175, c='r')
    # ax.set_xlabel('x[m]')
    # ax.set_ylabel('y[m]')

# fig.suptitle("Main Title")
fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.82, 0.31, 0.02, 0.37]) # [left, bottom, width, height] for 3 subplots
cbar_ax = fig.add_axes([0.84, 0.21, 0.02, 0.57])
fig.colorbar(im, cax=cbar_ax)

plt.show()
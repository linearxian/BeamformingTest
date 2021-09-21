# -*- coding: utf-8 -*-

from os import path

# imports from acoular

from acoular import __file__ as bpath, L_p, MicGeom, PowerSpectra,\
TimeSamples, MaskedTimeSamples, RectGrid3D, BeamformerBase, BeamformerCleansc, \
SteeringVector, WNoiseGenerator, PointSource, SourceMixer

# other imports
from numpy import mgrid, arange, array, arccos, pi, cos, sin, sum
import mpl_toolkits.mplot3d
from pylab import figure, show, scatter, subplot, imshow, title, colorbar,\
xlabel, ylabel


# micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
micgeofile = 'array_9.xml'

m = MicGeom( from_file=micgeofile )

ts = MaskedTimeSamples( name="44.h5", invalid_channels=[9] )

g = RectGrid3D(x_min=-0.2, x_max=4, 
               y_min=-0.2, y_max=4, 
               z_min=0, z_max=2, 
               increment=0.02)

# 50 hz for 1024 block size, because 51200/1024=50
# 40*50 - 160*50 for range 2000 hz to 8000 hz

f = PowerSpectra(time_data=ts, 
                 window='Hanning', 
                 overlap='50%', 
                 block_size=1024, 
                 ind_low=40, ind_high=160)
st = SteeringVector(grid=g, mics=m, steer_type='true location') 
b = BeamformerBase(freq_data=f, steer=st)

map = b.synthetic(5000,1)


fig=figure(1,(8,8))

# plot the results

subplot(221)
map_z = sum(map,2)
mx = L_p(map_z.max())
imshow(L_p(map_z.T), vmax=mx, vmin=mx-20, origin='lower', interpolation='nearest', 
       extent=(g.x_min, g.x_max, g.y_min, g.y_max))
xlabel('x')
ylabel('y')
title('Top view (xy)' )

subplot(223)
map_y = sum(map,1)
imshow(L_p(map_y.T), vmax=mx, vmin=mx-20, origin='upper', interpolation='nearest', 
       extent=(g.x_min, g.x_max, g.z_max, g.z_min))
xlabel('x')
ylabel('z')
title('Side view (xz)' )

subplot(222)
map_x = sum(map,0)
imshow(L_p(map_x), vmax=mx, vmin=mx-20, origin='lower', interpolation='nearest', 
       extent=(g.z_min, g.z_max, g.y_min, g.y_max))
xlabel('z')
ylabel('y')
title('Side view (zy)' )
colorbar()


# plot the setup

ax0 = fig.add_subplot((224), projection='3d')
ax0.scatter(m.mpos[0],m.mpos[1],m.mpos[2])
ax0.set_zlim(0,2) 
# source_locs=array([p1.loc,p2.loc,p3.loc]).T
# ax0.scatter(source_locs[0],source_locs[1],-source_locs[2])
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_zlabel('z')
ax0.set_title('Setup (mic and source positions)')

# only display result on screen if this script is run directly
if __name__ == '__main__': show()
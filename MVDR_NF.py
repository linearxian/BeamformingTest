import matplotlib.pyplot as plt
import numpy as np
from acoular import MicGeom, RectGrid
import math

def getNFSV(s_pos, mic_pos, freq, C = 343):
    lmbda = C / freq
    r_0p = math.dist(mic_pos[0], s_pos)
    r_mp = []
    a = []
    for i in np.arange(len(mic_pos)):
        r_ip = math.dist(mic_pos[int(i)], s_pos)
        delta_r_mp = r_0p - r_ip
        a_i = r_0p / r_ip * np.exp(-1 * 1j * 2 * np.pi * delta_r_mp / lmbda)
        r_mp.append(r_ip)
        # print(r_ip)
        # print(a_i)
        a.append(a_i)
    a = np.array(a).reshape([1,len(mic_pos)])
    return a

"""
parameters
"""
C = 343
f = 4000
lmbda = C / f
mg = MicGeom( from_file='array_9.xml' )
N = mg.mpos.shape[1]
mic_pos = []
for i in np.arange(N):
    mic_pos.append(mg.mpos[:,int(i)])

noise_coord = [4, 2, 1]
look_coord = [0, 2, 1]

"""
define signal
"""
x = getNFSV(noise_coord, mic_pos, f)
Rx = np.mat(np.outer(x, np.transpose(np.conj(x)))) # CSM
# plt.imshow(np.real(Rx), cmap='hot', interpolation='nearest')
# plt.show()

"""
define steering vector
"""
a_look = getNFSV(look_coord, mic_pos, f)
a_look = np.mat(a_look).T

"""
compute MVDR weights
"""
w = Rx.I * a_look / (a_look.H * Rx.I * a_look)

"""
define grid
"""
rg = RectGrid( x_min=-0.2, x_max=4,
                       y_min=-0.2, y_max=4,
                       z=1, increment=0.05 )

a_g = np.empty([rg.gpos.shape[1], len(mic_pos)], dtype=np.complex128)
for ind_g in np.arange(rg.gpos.shape[1]):
    a_column = []
    a_column = getNFSV(rg.gpos[:,ind_g], mic_pos, f)
    a_g[ind_g, :] = a_column

# print(a_g.shape)

"""
compute beamformer
"""
B = w.H * a_g.T
B = np.abs(B) / np.max(np.abs(B)) # normalization
# print(B.shape)

"""
plot beamformer
"""
fig = plt.figure()
ax3 = plt.axes(projection='3d')

X_g = rg.gpos[0,:].reshape(85,85)
Y_g = rg.gpos[1,:].reshape(85,85)
Z_g = B.reshape(85,85)

ax3.set_xlabel('$X (m)$')
ax3.set_ylabel('$Y (m)$')
ax3.set_zlabel('$Gain$')

ax3.plot_surface(X_g,Y_g,Z_g,cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()
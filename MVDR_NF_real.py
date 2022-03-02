import matplotlib.pyplot as plt
import numpy as np
from acoular import MicGeom, RectGrid
import math
import h5py
from scipy.fftpack import fft, ifft
from scipy import signal


def getSpectrogram(wav_data, frame, shift, fftl):
    len_sample, len_channel_vec = np.shape(wav_data)            
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    # window = sg.hanning(fftl + 1, 'periodic')[: - 1]   
    st = 0
    ed = frame
    number_of_frame = int((len_sample - frame) /  shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fftl, axis=1)[:, 0:int(fftl / 2) + 1] # channel * number_of_bin        
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums

def spec2wav(spectrogram, sampling_frequency, fftl, frame_len, shift_len):
    n_of_frame, fft_half = np.shape(spectrogram)    
    # hanning = sg.hanning(fftl + 1, 'periodic')[: - 1]    
    cut_data = np.zeros(fftl, dtype=np.complex64)
    result = np.zeros(sampling_frequency * 60 * 5, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]        
        cut_data[0:int(fftl / 2) + 1] = half_spec.T   
        cut_data[int(fftl / 2) + 1:] =  np.flip(np.conjugate(half_spec[1:int(fftl / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fftl))        
        result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2) 
        start_point = start_point + shift_len
        end_point = end_point + shift_len
    return result[0:end_point - shift_len]

def getNFSV(s_pos, mic_pos, freq, C = 343):
    if freq != 0:
        lmbda = C / freq
        r_0p = math.dist(mic_pos[0], s_pos)
        r_mp = []
        a = []
        for i in np.arange(len(mic_pos)):
            r_ip = math.dist(mic_pos[int(i)], s_pos)
            delta_r_mp = r_0p - r_ip
            a_i = r_0p / r_ip * np.exp(-1 * 1j * 2 * np.pi * delta_r_mp / lmbda)
            r_mp.append(r_ip)
            a.append(a_i)
    else:
        r_0p = math.dist(mic_pos[0], s_pos)
        r_mp = []
        a = []
        for i in np.arange(len(mic_pos)):
            r_ip = math.dist(mic_pos[int(i)], s_pos)
            a_i = r_0p / r_ip
            r_mp.append(r_ip)
            a.append(a_i)
    a = np.array(a).reshape([1,len(mic_pos)])
    weight = np.matmul(a, np.conjugate(a).T)
    a_normalized = a / weight
    return a_normalized

def applyBeamformer(beamformer, complex_spectrum):
    number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
    enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
    for f in range(0, number_of_bins):
        enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
    return enhanced_spectrum
    
"""
parameters
"""
mg = MicGeom( from_file='array_9.xml' )
number_of_mic = mg.mpos.shape[1]
mic_pos = []
for i in np.arange(number_of_mic):
    mic_pos.append(mg.mpos[:,int(i)])

# noise_coord = [4, 2, 0.5]
# look_coord = [0, 2, 0.5]
look_coord = [4, 2, 0.5]

"""
load signal
"""
fs = 51200
f = h5py.File('/media/xian/Data/dataset/Bekaert/h5files/42.h5', 'r')
dset = f['time_data']
source_signal = np.empty([dset.shape[0],dset.shape[1]-1], dtype=float)
for ind in np.arange(number_of_mic):
    source_signal[:,ind] = dset[:,ind]

f2 = h5py.File('/media/xian/Data/dataset/Bekaert/h5files/41.h5', 'r')
dset2 = f2['time_data']
multi_signal = np.empty([dset2.shape[0],dset2.shape[1]-1], dtype=float)
for ind in np.arange(number_of_mic):
    multi_signal[:,ind] = dset2[:,ind]

fft_length = 512
fft_shift = 256
use_number_of_frames_init = 10
use_number_of_frames_final = 10

frequency_grid = np.linspace(0, fs, fft_length)
frequency_grid = frequency_grid[0:int(fft_length / 2) + 1]
start_index = 0
end_index = start_index + fft_length
record_length, number_of_channels = np.shape(multi_signal)
R_mean = np.zeros((number_of_mic, number_of_mic, len(frequency_grid)), dtype=np.complex64)
used_number_of_frames = 0

# forward
for _ in range(0, use_number_of_frames_init):
    multi_signal_cut = multi_signal[start_index:end_index, :]
    complex_signal = fft(multi_signal_cut, n=fft_length, axis=0)
    for f in range(0, len(frequency_grid)):
            R_mean[:, :, f] = R_mean[:, :, f] + \
                np.outer(complex_signal[f, :], np.conjugate(complex_signal[f, :]).T)
    used_number_of_frames = used_number_of_frames + 1
    start_index = start_index + fft_shift
    end_index = end_index + fft_shift
    if record_length <= start_index or record_length <= end_index:
        used_number_of_frames = used_number_of_frames - 1
        break            

# backward
end_index = record_length
start_index = end_index - fft_length
for _ in range(0, use_number_of_frames_final):
    multi_signal_cut = multi_signal[start_index:end_index, :]
    complex_signal = fft(multi_signal_cut, n=fft_length, axis=0)
    for f in range(0, len(frequency_grid)):
        R_mean[:, :, f] = R_mean[:, :, f] + \
            np.outer(complex_signal[f, :], np.conjugate(complex_signal[f, :]).T)
    used_number_of_frames = used_number_of_frames + 1
    start_index = start_index - fft_shift
    end_index = end_index - fft_shift            
    if  start_index < 1 or end_index < 1:
        used_number_of_frames = used_number_of_frames - 1
        break                    

R_mean = R_mean / used_number_of_frames

tol = 1e-8
R_mean[abs(R_mean) < tol] = 0.0
# np.fill_diagonal(R_mean, 0) # diagonal removal

"""
define steering vector
"""
steering_vector = np.empty((len(frequency_grid), number_of_mic), dtype=np.complex64)
for f, frequency in enumerate(frequency_grid):
    steering_vector[f,:] = getNFSV(look_coord, mic_pos, frequency)

"""
compute MVDR weights
"""
beamformer = np.empty((number_of_mic, len(frequency_grid)), dtype=np.complex64)
for f in range(0, len(frequency_grid)):
    R_cut = np.reshape(R_mean[:, :, f], [number_of_mic, number_of_mic])
    R_cut = np.mat(R_cut)
    steering_vector_cut = steering_vector[f,:]
    steering_vector_cut = np.mat(steering_vector_cut).T
    beamformer[:, f] = np.asarray(R_cut.I * steering_vector_cut / (steering_vector_cut.H * R_cut.I * steering_vector_cut)).reshape(number_of_mic,)

"""
apply MVDR
"""
# frame = 512
# shift = 256
# number_of_frame = int((record_length - frame) /  shift) + 1
# complex_spectrum = np.empty((number_of_mic, len(frequency_grid), number_of_frame))
# for ind in np.arange(number_of_mic):
#     f, t, Sxx = signal.spectrogram(multi_signal[:,0], fs, nperseg=512, noverlap=256, mode='complex')
#     complex_spectrum[ind,:,:] = Sxx

complex_spectrum = getSpectrogram(source_signal, 512, 512, 512) # no window, no overlap
enhanced_spectrum = applyBeamformer(beamformer, complex_spectrum)
enhanced_audio = spec2wav(enhanced_spectrum, fs, 512, 512, 512)
enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.7

# time = np.linspace(0, enhanced_audio.shape[0] / fs, num=enhanced_audio.shape[0])
# plt.figure(1)
# plt.title("Signal Wave...")
# plt.plot(time, enhanced_audio)
# plt.show()

from scipy.io.wavfile import write
write("enhanced_audio.wav", fs, enhanced_audio.astype(np.float32))
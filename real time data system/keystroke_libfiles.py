'''general library'''
import msvcrt
import argparse
import threading
import queue
import sys
import time
import random
from matplotlib.animation import FuncAnimation
from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt
import numpy as np

'''audio library'''
import sounddevice as sd
import scipy.io.wavfile
from scipy import signal
import librosa

'''deep learning library'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

'''file library'''
# pickle library help write and load dictionary file, which help match label in cnn model
import pickle
import os
from os import listdir
from os.path import isfile, join



# audio section

def get_stft_param():
    '''
    This function provide parameter for STFT which used in almost all files
    '''
    dft_size = 256
    hop_size = int(1/4 * dft_size)
    zero_pad = dft_size
    window = np.hamming(dft_size)
    return dft_size, hop_size, zero_pad, window


def get_sr():
    '''
    This function provide a sample rate standard for subsequent processing
    '''
    return 44100



def window(size):
    '''
    This function create a average filter
    '''
    return np.ones(size)/float(size)
    

def smooth_filter(x_input, count):
    '''
    This funciton perform a convolution using average filter recursively 
    '''
    if count <= 0:
        return np.convolve(abs(x_input),window(100),'same')
    else:
        return np.convolve(smooth_filter(x_input,count-1),window(10),'same')
    

def fre_axis(sr_sound,x_sound,title):
    '''
    This function plot a time series
    '''
    fig, ax = plt.subplots(1,1)
    ax.plot(np.linspace(0,np.shape(x_sound)[0],np.shape(x_sound)[0]), x_sound)
    ax.set_title(title)


def fre_2axis(sr_sound,x_sound,title):
    fig, ax = plt.subplots(1,1)
    for line in x_sound:
        ax.plot(np.linspace(0,np.shape(line)[0],np.shape(line)[0]), line)
    ax.set_title(title)


def sound( x, rate=8000, label=''):
    '''
    This function play the given time series as audio. Only work in ipynb file.
    '''
    from IPython.display import display, Audio, HTML
    display( HTML( 
    '<style> table, th, td {border: 0px; }</style> <table><tr><td>' + label + 
    '</td><td>' + Audio( x, rate=rate)._repr_html_()[3:] + '</td></tr></table>'
    ))


def wavreadlocal(filename):
    '''
    This function load the data from an audio file
    '''
    import urllib.request, io, scipy.io.wavfile
    # f = wave.open(filename,"rb")
    sr,s = scipy.io.wavfile.read(filename)
    return sr, s.astype( 'float32')/32768


def wavwritelocal(filename, sr, data):
    '''
    This function use numpy array to write a wav file to loacl
    '''
    # Assuming data is in the range -1 to 1, denormalize to -32768 to 32767
    int_data = np.int16(data * 32767)

    # Write to WAV file
    scipy.io.wavfile.write(filename, sr, int_data)


def stft( input_sound, dft_size, hop_size, zero_pad, window):
    '''
    This function return frequency domain of a given time serires by Short-time Fourier transform
    '''
    dft_sound = np.lib.stride_tricks.sliding_window_view(input_sound, dft_size)[::hop_size, :]

    dft_sound = window*dft_sound

    if zero_pad:
        dft_sound = np.fft.rfft(dft_sound,dft_size+zero_pad)
    else:
        dft_sound = np.fft.rfft(dft_sound)
    
    return dft_sound

def overlap_add(idft_sound, hop_size, window):
    n_frames, frame_len = idft_sound.shape
    total_len = (n_frames - 1) * hop_size + frame_len
    result = np.zeros(total_len)

    for i, frame in enumerate(idft_sound):
        start = i * hop_size
        end = start + frame_len
        result[start:end] += frame * window

    return result

def istft(stft_output, dft_size, hop_size, zero_pad, window):
    idft_sound = np.fft.irfft(stft_output)
    idft_sound_without_padding = idft_sound[:, :dft_size]
    idft_sound_windowed = idft_sound_without_padding * window
    resynthesized_signal = overlap_add(idft_sound_windowed, hop_size, window)
    return resynthesized_signal


def stft_axis(sr_sound, x_sound, stft_sound, title):
    time_axis = np.linspace(0, len(x_sound) / sr_sound, num=np.shape(stft_sound)[0] + 1)
    freq_axis = np.fft.rfftfreq(np.shape(stft_sound)[1], 1 / sr_sound)
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.imshow(abs(np.power(stft_sound.T, 0.3)), origin="lower", aspect='auto', extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])

# end audio section



# data processing section

def find_thre_index(x, div):
    '''
    This function locate the start of keystroke by finding the magnitute which has max(x)/div
    e.g. if div = 2, then this function consider the first point that has half magnitute of the max value in the given data as the start of keystroke
    thus this function is unreliable when there is loud noise
    '''
    thre = np.max(x)/div

    thre_list_0 = np.argwhere(x[:,0]>thre)
    thre_list_1 = np.argwhere(x[:,1]>thre)

    if len(thre_list_0) <= 0:
        min_0 = float('inf')
    else:
        min_0 = thre_list_0[0]
    if len(thre_list_1) <= 0:
        min_1 = float('inf')
    else:
        min_1 = thre_list_1[0]

    # print(f"min_0:{min_0}")
    # print(f"min_1:{min_1}")

    thre_index = min(min_0,min_1)[0]

    return thre_index


def cut_length(x, length):
    '''
    This function change the length of x to given 'length' parameter by padding 0 to the end of x
    '''
    if len(x) < length:
        pad_rest = length - len(x)
        x = np.pad(x, (0,pad_rest), 'constant', constant_values=(0))
    
    return x


def cut_length_2c(x, length):
    '''
    This function change the length of x(shape:(data_len,2)) to shape:(length, 2) by padding 0 to x
    '''
    if len(x[:,0]) < length:
        pad_rest = length - np.shape(x)[0]
        x = np.pad(x, ((0,pad_rest),(0,0)), 'constant', constant_values=(0))

    return x


def locate_noise_maxdiv(x, div, length):
    '''
    This function locate the keystroke based on the max value in the given x(shape:(data_len,2))
    It locate the keystroke by using find_thre_index to find the start of keystroke, then use cut_length_2c to ensure the keystroke's length equal to 'length'
    '''
    thre_index = find_thre_index(x, div)
    left_cut = int(length*(1/9))
    keystroke = x[thre_index-length-left_cut: thre_index-left_cut, :]
    keystroke_x = cut_length_2c(keystroke, length)

    return keystroke_x


def locate_keystroke_maxdiv(x, div, length):
    '''
    This function locate the keystroke based on the max value in the given x(shape:(data_len,2))
    It locate the keystroke by using find_thre_index to find the start of keystroke, then use cut_length_2c to ensure the keystroke's length equal to 'length'
    '''
    thre_index = find_thre_index(x, div)
    left_cut = int(length*(1/9))
    keystroke = x[thre_index-left_cut: thre_index+length-left_cut, :]
    keystroke_x = cut_length_2c(keystroke, length)

    return keystroke_x


def locate_keystroke_rnn(x, div, length):
    '''
    This function locate the keystroke based on the trained RNN network
    not implement yet
    '''
    pass


def preprocessing_data(audio_x, prepro_type):
    '''
    This function receive audio_x(shape:(data_len, 2)) and convert it into preprocessed data type e.g. stft, mfcc
    '''
    processed_data = None

    # preprocess data by stft
    if prepro_type == "stft":
        dft_size, hop_size, zero_pad, stft_window = get_stft_param()
        audio_x = audio_x.T
        stft_list = []
        for channel in audio_x:
            # print(f"channel.shape:{channel.shape}")
            stft_list.append(abs(stft( channel, dft_size, hop_size, zero_pad, stft_window)))
        processed_data = stft_list

    # preprocess data by mfcc
    elif prepro_type == "mfcc":
        # print("mfcc is not tseted yet, please choose another way to preprocess data")
        sr = get_sr()
        audio_x = audio_x.T
        mfcc_list = []
        for channel in audio_x:
            mfcc_list.append(librosa.feature.mfcc(y=channel, sr=sr))
        processed_data = mfcc_list

    return processed_data


# end data processing section



# deep learning section

class Net(nn.Module):
        '''
        The cnn structure that used in all steps
        Must include it when train, save, load, and use model
        '''
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(2, 16, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            '''stft'''
            self.fc1 = nn.Linear(69632, 128)
            '''mfcc'''
            # self.fc1 = nn.Linear(640, 128)
            self.fc2 = nn.Linear(128, 29)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


def get_label_29keys():
    '''
    This function return two ductionary:
    1. dic_ktol dictionary with keyboard button as key, number label as value
    2. dic_ltot ictionary with number label as key, keyboard button as value
    dic_ktol is used when number label is needed for CNN network
    dic_ltot is used when we want to know corresponding keyboard button result from CNN network's output.
    '''
    dic_ktol = {}
    dic_ltot = {}
    count = 0
    for letter in "abcdefghijklmnopqrstuvwxyz,. ":
        dic_ktol[letter] = count
        dic_ltot[count] = letter
        count += 1
    return dic_ktol, dic_ltot

# end deep learning section
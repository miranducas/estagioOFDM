#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
import statistics as sta
from cmath import sqrt
from cmath import pi
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from scipy.linalg import norm
from sklearn import preprocessing
from scipy.linalg import toeplitz
from scipy.linalg import circulant
import math 
import pickle
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy import ndarray
################################################################################################################################
# This script 
################################################################################################################################
#----------------------------------------------------------------------------------------------------------------
# Subcarrier Allocation
#----------------------------------------------------------------------------------------------------------------
def SubcarrierAllocation(S,num_pilot,random_pilot):
    allCarriers = np.arange(S)  # indices of all subcarriers ([0, 1, ... K-1])
    if random_pilot == True:
        pilotCarriers = np.arange(1,S-1,1)
        random.shuffle(pilotCarriers)
        pilotCarriers = pilotCarriers[0:num_pilot-2]
        pilotCarriers.sort()
        pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
        pilotCarriers = np.hstack([np.array([allCarriers[0]]), pilotCarriers])
    else:
        all_aux = allCarriers[0:S-1]
        pilotCarriers = all_aux[::(S-1)//(num_pilot-1)] 
        pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    # data carriers are all remaining carriers
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    # print ("allCarriers:   %s" % allCarriers)
    # print ("pilotCarriers: %s" % pilotCarriers)
    # print ("dataCarriers:  %s" % dataCarriers)

    # plt.figure(figsize=(8,0.8))
    # plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
    # plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
    # plt.legend(fontsize=10, ncol=2)
    # plt.xlim((-1,S)); plt.ylim((-0.1, 0.3))
    # plt.xlabel('Carrier index')
    # plt.yticks([])
    # plt.grid(True);
    # plt.show() 
    return allCarriers,pilotCarriers,dataCarriers

#----------------------------------------------------------------------------------------------------------------
# S/P converter
#----------------------------------------------------------------------------------------------------------------
def SP(bits,dataCarriers,mu):
    return bits.reshape(((dataCarriers), mu))


#----------------------------------------------------------------------------------------------------------------
# Mapping bits to symbols
#----------------------------------------------------------------------------------------------------------------
def Mapping(bits,mapping_table):
    return np.array([mapping_table[tuple(b)] for b in bits])


#----------------------------------------------------------------------------------------------------------------
# Allocation of sub-carriers
#----------------------------------------------------------------------------------------------------------------
def OFDM_symbol(QAM_payload,S,pilotValue,pilotCarriers,dataCarriers):
    symbol = np.zeros(S, dtype=complex) # the overall S subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the data subcarriers
    return symbol


def OFDM_symbol_all_sub(S,QAM_payload):
    dataCarriers = np.arange(S)
    symbol = np.zeros(S, dtype=complex) # overall S subcarriers
    symbol[dataCarriers] = QAM_payload # allocate the data subcarriers
    return symbol
#----------------------------------------------------------------------------------------------------------------
# Conversion to time-domain
#----------------------------------------------------------------------------------------------------------------
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

#----------------------------------------------------------------------------------------------------------------
# Redundancy add
#----------------------------------------------------------------------------------------------------------------

def add_redund(OFDM_time, K, N, type):
    if K==0 :
        OFDM_TX = OFDM_time
    else:
        if type == 'CP':
            cp = OFDM_time[-K:]  # take the last CP samples
            OFDM_TX = np.hstack([cp, OFDM_time])

        if type == 'ZP':
            F = np.vstack((np.eye(N), np.zeros((K, N))))
            OFDM_TX = np.dot(F, OFDM_time)
        if type == 'ZP-SC':
            F = np.vstack((np.eye(N), np.zeros((K, N))))
            OFDM_TX = np.dot(F, OFDM_time)
        if type == 'CP-SC':
            aux1 = np.zeros((K,N-K))
            aux2 = np.eye(K)
            aux = np.hstack((aux1,aux2))
            F = np.vstack((aux, np.eye(N)))
            OFDM_TX = np.dot(F, OFDM_time)

    return OFDM_TX  # add the Prefix to the beginning

#----------------------------------------------------------------------------------------------------------------
# Addition of noise and multiplication with channel
#----------------------------------------------------------------------------------------------------------------
def channel(signal,signal_ant,channelResponse,S,SNRdb):
    A = np.zeros(shape=(S,S),dtype=np.complex)
    hh = channelResponse[1:len(channelResponse)]
    HH = toeplitz(hh[::-1])
    T=np.triu(HH)
    for s in range(len(hh)):
        A[s] = np.append(np.zeros(shape=(1,S-len(hh)),dtype=np.complex),T[s])
    
    h = np.append(channelResponse,np.zeros(S-len(channelResponse)))
    H = circulant(h)
    convolved = np.dot((H-A),signal) + np.dot((A),signal_ant)
    
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10) # calculate noise power based on signal power and SNR

    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape) + 1j*np.random.randn(*convolved.shape))     
    return (convolved + noise), sigma2 

#----------------------------------------------------------------------------------------------------------------
# remove redundancy
#----------------------------------------------------------------------------------------------------------------

def remove_redund(OFDM_RX, K, N, L, type):

    if K == 0:
        result = OFDM_RX
    else:
        if type == 'CP':
            result = OFDM_RX[K:(K+N)]

        if type == 'ZP-OLA':
            G_barra = np.eye(K)
            ii = np.zeros((N - K, K))
            g_aux = np.vstack((G_barra, ii))
            G = np.hstack((np.eye(N), g_aux))
            result = np.dot(G,OFDM_RX)
        if type == 'ZPZJ':
            result = OFDM_RX[(L - K):(N+K)]
        if type == 'ZP-SC':
            G_barra = np.eye(K)
            ii = np.zeros((N - K, K))
            g_aux = np.vstack((G_barra, ii))
            G = np.hstack((np.eye(N), g_aux))
            result = np.dot(G,OFDM_RX)            
        if type == 'CP-SC':
            G = np.hstack((np.zeros((N,K)),np.eye(N)))
            result = np.dot(G,OFDM_RX) 
            
    return result

#----------------------------------------------------------------------------------------------------------------
#Transform to frequency domain
#----------------------------------------------------------------------------------------------------------------
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

#----------------------------------------------------------------------------------------------------------------
# Channel estimation
#----------------------------------------------------------------------------------------------------------------
def channelEstimate_all_pilot_sub(OFDM_demod,pilotCarriers,pilotValue):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    return Hest_at_pilots

def channelEstimate(OFDM_demod,allCarriers,pilotCarriers,pilotValue):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
       
    return Hest

def channelEstimate_exact(Hest_at_pilots,allCarriers,pilotCarriers):
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
       
    return Hest


def MMSEchannelEstimate(OFDM_demod,pilotValue, Nfft, Nps, h, SNR):
    snr = 10**(SNR*0.1)
    Np = round(Nfft)
    Y = OFDM_demod
    H_tilde = Y/(pilotValue)

    k = np.array(range(len(h)))
    hh = np.dot(h,h.T)
    tmp = h*np.conj(h)*k
    
    r = sum(tmp)/hh
    r2 = np.dot(tmp,(k))/hh
    tau_rms = sqrt(r2-r**2)
    df = 1/Nfft
    j2pi_tau_df = 1j*2*pi*tau_rms*df
    K1 = np.tile(np.arange(Np).reshape(-1,1),(1,Np))
    K2 = np.tile(np.arange(Np),(Np,1))
    rf = 1/(1+j2pi_tau_df*Nps*(K1-K2))
    
    K3 = np.tile(np.arange(Np).reshape(-1,1),(1,Np))
    K4 = np.tile(np.arange(Np),(Np,1))   
    rf2 = 1/(1+j2pi_tau_df*Nps*(K3-K4))
    Rhp = rf
    Rpp = rf2 + (1/snr)*np.eye(len(H_tilde), dtype=int)
    
    Hest_at_pilots = np.dot(np.dot(Rhp,np.linalg.inv(Rpp)),H_tilde.flatten())
    

    return Hest_at_pilots

def pesos(Y, Xp, pilot_loc, Nfft, Nps, h, SNR):
    snr = 10**(SNR*0.1)
    Np = round(Nfft)
    H_tilde = Y/Xp
    k = np.array(range(len(h)))
    hh = np.dot(h,h)
    tmp = h*np.conj(h)*k
    
    r = sum(tmp)/hh
    r2 = np.dot(tmp,(k))/hh
    tau_rms = sqrt(r2-r**2)
    df = 1/Nfft
    j2pi_tau_df = 1j*2*pi*tau_rms*df
    K1 = np.tile(np.arange(Np).reshape(-1,1),(1,Np))
    K2 = np.tile(np.arange(Np),(Np,1))
    rf = 1/(1+j2pi_tau_df*Nps*(K1-K2))
    
    K3 = np.tile(np.arange(Np).reshape(-1,1),(1,Np))
    K4 = np.tile(np.arange(Np),(Np,1))   
    rf2 = 1/(1+j2pi_tau_df*Nps*(K3-K4))
    Rhp = rf
    Rpp = rf2 + (1/snr)*np.eye(len(H_tilde), dtype=int)
    
    W_pesos = np.dot(Rhp,np.linalg.inv(Rpp))
    
    return W_pesos

#----------------------------------------------------------------------------------------------------------------
# Convert to block of real and imag
#----------------------------------------------------------------------------------------------------------------
def block_real_imag(H_1):
    H_LS = H_1
    H_LS_real = H_LS.real
    H_LS_imag = H_LS.imag
    H_LS_block = np.append(H_LS_real,H_LS_imag)
    return H_LS_block

#----------------------------------------------------------------------------------------------------------------
# Demapping QAM symbols
#----------------------------------------------------------------------------------------------------------------
def Demapping(QAM,demapping_table):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision])

#----------------------------------------------------------------------------------------------------------------
# Frequency Domain Equalizer (FDE) 
#----------------------------------------------------------------------------------------------------------------
def equalizer(OFDM_demod, Hest):
    return OFDM_demod/Hest

#----------------------------------------------------------------------------------------------------------------
# Payload Extraction
#----------------------------------------------------------------------------------------------------------------
def get_payload(equalized,dataCarriers):
    return equalized[dataCarriers]

#----------------------------------------------------------------------------------------------------------------
# P/S conversion
#----------------------------------------------------------------------------------------------------------------
def PS(bits):
    return bits.reshape((-1,))


def inversa_MMSE(channelResponse, S, cp, SNRdb):
    L = len(channelResponse) - 1
    N = S - cp
    A = np.zeros(shape=(S, S), dtype=np.complex)
    hh = channelResponse[0:len(channelResponse)]
    HH = toeplitz(hh[::-1])
    T = np.triu(HH)

    for k in range(len(hh)):
        A[k] = np.append(np.zeros(shape=(1, S - len(hh)), dtype=np.complex), T[k])

    h = np.append(channelResponse, np.zeros(S - len(channelResponse)))
    H = circulant(h)
    F_barra = np.eye(N)
    F = np.vstack((F_barra, np.zeros((cp, N))))
    if cp == L:
        H_barra = np.dot((H - A), F)
        matrixx = np.dot(np.conj(np.transpose(H_barra)), H_barra)
    else:
        H_barra = np.dot((H - A), F)
        H_barra = H_barra[(L - cp):S, :]
        # import pdb
        # pdb.set_trace()
        # matrixx = np.dot(np.conj(np.transpose(H_barra)), H_barra) + 10 ** (-SNRdb / 10) * np.eye(N)
        matrixx = np.dot(np.conj(np.transpose(H_barra)), H_barra) + 10 ** (-3) * np.eye(N)


    matrix = np.dot(inv(matrixx), np.conj(np.transpose(H_barra)))
    # aux_inv = np.hstack((np.zeros((N,L-cp)),matrix))

    return (matrix)

def channel_winner(taps):
    # OR = 4; 
    Mm = 256
    # Dop_res = 0.1; 
    # res_accu = 20;
    l_ch = 1
    s_channel = np.random.binomial(1, 0.5, 1)
    #C1 urban micro-cell
    if s_channel==1:
        Pp = np.array([-1,-3.2,-5,-7.5,-10.5,-3.2,-6.1,-8.3,-14.1,-18,-23.2,-23.1,-24.6,-26,-27.2,-29.1,-29.5,-31.4,-29.9])
    else:
        Pp = np.array([0,-25.3,-27.1,-21.6,-26.3,-25.1,-25.4,-22.0,-29.2,-24.3,-26.5,-28.2,-23.2,-32.2,-26.5,-32.1,-28.5,-30.5,-32.6])

    Kk = np.concatenate((np.array([ 1, 1, 1 ]),np.array(np.ones(16)))) 
    # tau = np.array([ 0.0, 0.5, 1.0, 2 , 3 , 4 , 5 , 6 , 7 , 8, 9,10,11,12,13,14,15,16,17 ])
    Dop = np.array(2.5*np.ones(taps))    
    # ant_corr = 0.3; 
    Fnorm = -1.5113

    Pp = 10**(Pp/10)
    s2 = Pp/(Kk+1)
    m2 = Pp*(Kk/(Kk+1))
    m = np.sqrt(m2)
    L = len(Pp)
    bun=np.sqrt(s2)
    paths_r = (np.sqrt(1/2)*(np.random.randn(L,l_ch) + 1j*np.random.randn(L,l_ch)))*(np.dot(np.reshape(bun, (len(bun), 1)),np.ones(shape=(1,l_ch))))
    paths_c = np.dot(np.reshape(m,(len(m),1)), np.ones(shape=(1,l_ch)))
    
    for p in range(L):
        D = Dop[p] / max(Dop) / 2
        f0 = np.arange(Mm*D+1)/(Mm*D)
        PSD = 0.785*f0**4 - 1.72*f0**2 + 1.0
        filt = np.concatenate((PSD[0:len(PSD)-1],PSD[:0:-1]))
        filt = np.sqrt(filt)
        filt = np.fft.ifftshift(filt)
        filt = filt.real
        filt = filt / np.sqrt(sum(filt**2))
        aux=paths_r[p]
        aux = np.append(aux,[0])
        path = np.convolve(filt, aux,'full')
        paths_r[p] = path[128:(128+l_ch)]
    
    paths = paths_r + paths_c
    paths = paths * 10**(Fnorm/20)
    
    ppp = np.reshape(paths,(taps,))
    return ppp


def model_estimate(model,y,N):
    y_block=(block_real_imag(y))
    y_pred = model.predict(np.reshape(y_block,(1,2*N)))
    y_est = []
    for i in range(N):
        y_est.append(y_pred[0,i]+1j*y_pred[0,i+N])
    y_est = np.asarray(y_est)
    return y_est

def model_estimate_norm(model,y,N,bits_per_symbol):
    number_of_symbols= 2**bits_per_symbol
    if number_of_symbols==16:
        denorm_factor = 27

    if number_of_symbols==64:
        denorm_factor = 50

    y_mr=(block_real_imag(y))
    y_block = np.reshape(y_mr,(1,-1))
    y_block = preprocessing.normalize(y_block)
    y_pred = model.predict(np.reshape(y_block,(1,2*N)))
    y_est = []
    for i in range(N):
        y_est.append(y_pred[0,i]+1j*y_pred[0,i+N])
    y_est = np.asarray(y_est)*denorm_factor
    return y_est

def Wmatrix(W):
    W_real=W.real
    W_imag=W.imag
    ww_1=np.concatenate((W_real,W_imag))
    ww_2=np.concatenate((W_imag,W_real))
    ww = np.concatenate((ww_1,ww_2),axis=1)
    return ww
#----------------------------------------------------------------------------------------------------------------
# QAM Modulation
#----------------------------------------------------------------------------------------------------------------
def QAM_mod(number_of_symbols):
    if number_of_symbols==16:
        map_table = {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -3-1j,
        (0,0,1,0) : -3+3j,
        (0,0,1,1) : -3+1j,
        (0,1,0,0) : -1-3j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : -1+3j,
        (0,1,1,1) : -1+1j,
        (1,0,0,0) : 3-3j,
        (1,0,0,1) : 3-1j,
        (1,0,1,0) : 3+3j,
        (1,0,1,1) : 3+1j,
        (1,1,0,0) : 1-3j,
        (1,1,0,1) : 1-1j,
        (1,1,1,0) : 1+3j,
        (1,1,1,1) : 1+1j        
        }


        demap_table = {v : k for k, v in map_table.items()}

    if number_of_symbols==64:
        map_table = {
        (0,0,0,0,0,0) : 3+3j,
        (0,0,0,0,0,1) : 3+1j,
        (0,0,0,0,1,0) : 1+3j,
        (0,0,0,0,1,1) : 1+1j,
        (0,0,0,1,0,0) : 3+5j,
        (0,0,0,1,0,1) : 3+7j,
        (0,0,0,1,1,0) : 1+5j,
        (0,0,0,1,1,1) : 1+7j,
        (0,0,1,0,0,0) : 5+3j,
        (0,0,1,0,0,1) : 5+1j,
        (0,0,1,0,1,0) : 7+3j,
        (0,0,1,0,1,1) : 7+1j,
        (0,0,1,1,0,0) : 5+5j,
        (0,0,1,1,0,1) : 5+7j,
        (0,0,1,1,1,0) : 7+5j,
        (0,0,1,1,1,1) : 7+7j,   
        (0,1,0,0,0,0) : 3-3j,        
        (0,1,0,0,0,1) : 3-1j,        
        (0,1,0,0,1,0) : 1-3j,        
        (0,1,0,0,1,1) : 1-1j,        
        (0,1,0,1,0,0) : 3-5j,        
        (0,1,0,1,0,1) : 3-7j,        
        (0,1,0,1,1,0) : 1-5j,        
        (0,1,0,1,1,1) : 1-7j,        
        (0,1,1,0,0,0) : 5-3j,        
        (0,1,1,0,0,1) : 5-1j,        
        (0,1,1,0,1,0) : 7-3j,        
        (0,1,1,0,1,1) : 7-1j,        
        (0,1,1,1,0,0) : 5-5j,        
        (0,1,1,1,0,1) : 5-7j,        
        (0,1,1,1,1,0) : 7-5j,        
        (0,1,1,1,1,1) : 7-7j,        
        (1,0,0,0,0,0) : -3+3j,        
        (1,0,0,0,0,1) : -3+1j,        
        (1,0,0,0,1,0) : -1+3j,        
        (1,0,0,0,1,1) : -1+1j,        
        (1,0,0,1,0,0) : -3+5j,        
        (1,0,0,1,0,1) : -3+7j,        
        (1,0,0,1,1,0) : -1+5j,        
        (1,0,0,1,1,1) : -1+7j,        
        (1,0,1,0,0,0) : -5+3j,        
        (1,0,1,0,0,1) : -5+1j,        
        (1,0,1,0,1,0) : -7+3j,        
        (1,0,1,0,1,1) : -7+1j,        
        (1,0,1,1,0,0) : -5+5j,        
        (1,0,1,1,0,1) : -5+7j,        
        (1,0,1,1,1,0) : -7+5j,        
        (1,0,1,1,1,1) : -7+7j,        
        (1,1,0,0,0,0) : -3-3j,        
        (1,1,0,0,0,1) : -3-1j,        
        (1,1,0,0,1,0) : -1-3j,        
        (1,1,0,0,1,1) : -1-1j,        
        (1,1,0,1,0,0) : -3-5j,        
        (1,1,0,1,0,1) : -3-7j,  
        (1,1,0,1,1,0) : -1-5j,
        (1,1,0,1,1,1) : -1-7j,
        (1,1,1,0,0,0) : -5-3j,
        (1,1,1,0,0,1) : -5-1j,
        (1,1,1,0,1,0) : -7-3j,
        (1,1,1,0,1,1) : -7-1j,
        (1,1,1,1,0,0) : -5-5j,
        (1,1,1,1,0,1) : -5-7j,
        (1,1,1,1,1,0) : -7-5j,
        (1,1,1,1,1,1) : -7-7j,
        }
        demap_table = {v : k for k, v in map_table.items()}


    return map_table,demap_table
    ################################################################################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:50:02 2020

@author: marcele
"""

#!/usr/bin/env python
# coding: utf-8
import numpy as np
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
import tensorflow.keras as keras
import tensorflow as tf
import scipy.signal as sg
from scipy.signal import butter, filtfilt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
import math 
import pickle
import ofdm
import ofdmclass
import ofdm_func
import pdb
from sklearn.preprocessing import StandardScaler
import scipy.io as spio


################################################################################################################################
# This script is used to obtain the average bit-error-rate (BER) for the algorithms considered in:
# "OFDM Receiver Using Deep Learning: Redundancy Issues"
# 
# The OFDM frame consists of one pilot and one data OFDM symbols, that is, the subcarriers are formed 
# by pilot symbols or data symbols only.
################################################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------
# System Parameters Decription
#-------------------------------------------------------------------------------------------------------------------------------
S = 64                         # Number of OFDM sub-carriers
SNRdb=0                    # Signal-to-noise ratio (value)
ensemble = 100                     # Number of independent runs to average    
mu = 4                              # Number of bits per symbol
QAM = 2**mu                       
train = False                     # True if training will be performed
nonlinearity = False            # True if nonlinearity is added at TX
channel_type = 'PED'            # 'PED' or 'VEH'
TX_type = 'CP-SC'
sc_flag = True
#-------------------------------------------------------------------------------------------------------------------------------
if channel_type=='PED':
    Vchannel = spio.loadmat('channel_ped.mat')
    channel_3g_train = Vchannel['y_ped_train']
    channel_3g_test = Vchannel['y_ped_test']
if channel_type=='VEH':
    Vchannel = spio.loadmat('channel_veh.mat')
    channel_3g_train = Vchannel['y_veh_train']
    channel_3g_test = Vchannel['y_veh_test']
if channel_type =='VEHPED':
    print('canal VEHPED')
    Vchannel = spio.loadmat('channel_vehped.mat')
    channel_3g_train = Vchannel['y_vehped_train']
    channel_3g_test = Vchannel['y_vehped_test']

taps = channel_3g_test.shape[1]
CP = taps-1
cp_met = CP//2

if nonlinearity == True:
    str_linear = 'nao_linear/'
if nonlinearity == False:
    str_linear = 'linear/'


[mapping_table,demapping_table] = ofdm.QAM_mod(2**mu)


def data_set_net1_zp(H_LS, H_true, channel_3g, size_data, SNRdb, cpp, K, nonlinearity):
    for x in range(size_data):
        # Canal gerado:
        channelResponse = channel_3g[x]
        H_exact = np.fft.fft(channelResponse, K)

        # Gerando pilotos a serem transmitidos e recebidos:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP = ofdm.SP(bits, K, mu)
        QAM = ofdm.Mapping(bits_SP, mapping_table)
        OFDM_data = ofdm.OFDM_symbol_all_sub(K, QAM)
        OFDM_time = ofdm.IDFT(OFDM_data)

        bits_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_ant = ofdm.SP(bits_ant, K, mu)
        QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
        OFDM_data_ant = ofdm.OFDM_symbol_all_sub(K, QAM_ant)
        OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)

        OFDM_TX = ofdm.add_redund(OFDM_time, cpp, K, 'ZP')
        OFDM_TX_ant = ofdm.add_redund(OFDM_time_ant, cpp, K, 'ZP')

        if nonlinearity == True:
            # OFDM_TX = np.tanh(OFDM_TX)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX))
            CL = CR * sigma_x
            OFDM_TX = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX]
            sigma_x_ant = np.sqrt(np.var(OFDM_TX_ant))
            CL_ant = CR * sigma_x_ant
            OFDM_TX_ant = [i / abs(i) * CL_ant if abs(i) > CL_ant else i for i in OFDM_TX_ant]

        [OFDM_RX, sigma] = ofdm.channel(OFDM_TX, OFDM_TX_ant, channelResponse, K + cpp, SNRdb)
        noise = np.sqrt(sigma / 2) * (
                    np.random.randn(*channelResponse.shape) + 1j * np.random.randn(*channelResponse.shape))
        channelResponse1 = channelResponse + noise
        H_exact = np.fft.fft(channelResponse1, K)

        OFDM_RX_ok = ofdm.remove_redund(OFDM_RX, cpp, K, len(channelResponse)-1, 'ZP-OLA')

        OFDM_demod = ofdm.DFT(OFDM_RX_ok)

        H_LS.append(ofdm.block_real_imag(OFDM_demod / OFDM_data))
        H_true.append(ofdm.block_real_imag(H_exact))

    X_train = np.asarray(H_LS)
    Y_train = np.asarray(H_true)
    return (X_train, Y_train)


def data_set_net1_CPSC(H_LS, H_true, channel_3g, size_data, SNRdb, cpp, K, nonlinearity):    
    for x in range(size_data):
        # Canal gerado:
        channelResponse = channel_3g[x]
        H_exact = np.fft.fft(channelResponse, K)

        # Gerando pilotos a serem transmitidos e recebidos:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP = ofdm.SP(bits, K, mu)
        QAM = ofdm.Mapping(bits_SP, mapping_table)
        OFDM_data = ofdm.OFDM_symbol_all_sub(K, QAM)
         
        bits_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_ant = ofdm.SP(bits_ant, K, mu)
        QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
        OFDM_data_ant = ofdm.OFDM_symbol_all_sub(K, QAM_ant)
            
        if sc_flag == True:
            OFDM_time = OFDM_data
            OFDM_time_ant = OFDM_data_ant
        else:
            OFDM_time = ofdm.IDFT(OFDM_data)
            OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)

        OFDM_TX = ofdm.add_redund(OFDM_time, cpp, K, 'CP-SC')
        OFDM_TX_ant =ofdm.add_redund(OFDM_time_ant, cpp, K, 'CP-SC')
        if nonlinearity==True:
            # OFDM_TX = np.tanh(OFDM_TX)
            # OFDM_TX_ant = np.tanh(OFDM_TX_ant)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX))
            CL = CR * sigma_x
            OFDM_TX = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX]
            CR = 1.3
            sigma_x = np.sqrt(np.var(OFDM_TX_ant))
            CL = CR * sigma_x
            OFDM_TX_ant = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX_ant]
            
        [OFDM_RX,sigma] = ofdm.channel(OFDM_TX,OFDM_TX_ant,channelResponse,K + cpp,SNRdb)
        noise = np.sqrt(sigma / 2) * (np.random.randn(*channelResponse.shape) + 1j * np.random.randn(*channelResponse.shape))
        channelResponse1 = channelResponse + noise
        H_exact = np.fft.fft(channelResponse1, K)

        OFDM_RX_ok = ofdm.remove_redund(OFDM_RX, cpp, K, len(channelResponse)-1, 'CP-SC')
       
        OFDM_demod = ofdm.DFT(OFDM_RX_ok)
        OFDM_data = ofdm.DFT(OFDM_data)

        Hest_ls =(OFDM_demod/(OFDM_data+10**(-8)))
       
        H_LS.append(ofdm.block_real_imag(Hest_ls))
        H_true.append(ofdm.block_real_imag(H_exact))
    X_train = np.asarray(H_LS)
    Y_train = np.asarray(H_true)
    return (X_train, Y_train)



def data_set_CE_net_min_red(H_LS, H_true, channel_3g, size_data, SNRdb, cpp, K, nonlinearity):
    for x in range(size_data):
        # Canal gerado:
        channelResponse = channel_3g[x]
        H_exact = np.fft.fft(channelResponse, K)

        # Gerando pilotos a serem transmitidos e recebidos:
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP = ofdm.SP(bits, K, mu)
        QAM = ofdm.Mapping(bits_SP, mapping_table)
        OFDM_data = ofdm.OFDM_symbol_all_sub(K, QAM)
        OFDM_time = ofdm.IDFT(OFDM_data)

        bits_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_ant = ofdm.SP(bits_ant, K, mu)
        QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
        OFDM_data_ant = ofdm.OFDM_symbol_all_sub(K, QAM_ant)
        OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)

        OFDM_TX = ofdm.add_redund(OFDM_time, cpp, K, 'ZP')
        OFDM_TX_ant = ofdm.add_redund(OFDM_time_ant, cpp, K, 'ZP')

        if nonlinearity == True:
            # OFDM_TX = np.tanh(OFDM_TX)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX))
            CL = CR * sigma_x
            OFDM_TX = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX]
            sigma_x_ant = np.sqrt(np.var(OFDM_TX_ant))
            CL_ant = CR * sigma_x_ant
            OFDM_TX_ant = [i / abs(i) * CL_ant if abs(i) > CL_ant else i for i in OFDM_TX_ant]

        [OFDM_RX, sigma] = ofdm.channel(OFDM_TX, OFDM_TX_ant, channelResponse, K + cpp, SNRdb)
        noise = np.sqrt(sigma / 2) * (
                    np.random.randn(*channelResponse.shape) + 1j * np.random.randn(*channelResponse.shape))
        channelResponse1 = channelResponse + noise
        H_exact = np.fft.fft(channelResponse1, K)

        OFDM_RX_ok = ofdm.remove_redund(OFDM_RX, cpp, K, len(channelResponse)-1, 'ZP-OLA')

        OFDM_demod = ofdm.DFT(OFDM_RX_ok)
        if x == 0:
            W = ofdm.pesos(OFDM_demod, OFDM_data, np.arange(K), K, K, channelResponse, SNRdb)
        H_LS.append(ofdm.block_real_imag(OFDM_demod / OFDM_data))
        H_true.append(ofdm.block_real_imag(H_exact))

    X_train = np.asarray(H_LS)
    Y_train = np.asarray(H_true)
    W_real = W.real
    W_imag = W.imag
    ww_1 = np.concatenate((W_real, W_imag))
    ww_2 = np.concatenate((W_imag, W_real))
    ww = np.concatenate((ww_1, ww_2), axis=1)
    return (X_train, Y_train, ww)

#--------------------------------------------------------------------------------------------------------------
def data_set_zpzj(y, y_true, channel_3g, size_data, model2, SNRdb, cpp, K, nonlinearity):
    for x in range(size_data):

        # Canal gerado:
        channelResponse = channel_3g[x]
        H_exact = np.fft.fft(channelResponse, K)

        # Gerando pilotos a serem transmitidos e recebidos:
        bits_piloto = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_piloto = ofdm.SP(bits_piloto, K, mu)
        QAM_piloto = ofdm.Mapping(bits_SP_piloto, mapping_table)
        OFDM_data_piloto = ofdm.OFDM_symbol_all_sub(K, QAM_piloto)
        OFDM_time_piloto = ofdm.IDFT(OFDM_data_piloto)

        bits_piloto_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_piloto_ant = ofdm.SP(bits_piloto_ant, K, mu)
        QAM_piloto_ant = ofdm.Mapping(bits_SP_piloto_ant, mapping_table)
        OFDM_data_piloto_ant = ofdm.OFDM_symbol_all_sub(K, QAM_piloto_ant)
        OFDM_time_piloto_ant = ofdm.IDFT(OFDM_data_piloto_ant)

        OFDM_TX_piloto = ofdm.add_redund(OFDM_time_piloto, cpp, K, 'ZP')
        OFDM_TX_piloto_ant = ofdm.add_redund(OFDM_time_piloto_ant, cpp, K, 'ZP')


        if nonlinearity == True:
            # OFDM_TX_piloto = np.tanh(OFDM_TX_piloto)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX_piloto))
            CL = CR * sigma_x
            OFDM_TX_piloto = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX_piloto]
            sigma_x_ant = np.sqrt(np.var(OFDM_TX_piloto_ant))
            CL_ant = CR * sigma_x_ant
            OFDM_TX_piloto_ant = [i / abs(i) * CL_ant if abs(i) > CL_ant else i for i in OFDM_TX_piloto_ant]

        [OFDM_RX_piloto, sigma] = ofdm.channel(OFDM_TX_piloto, OFDM_TX_piloto_ant, channelResponse, K + cpp, SNRdb)

        OFDM_RX_ok_piloto = ofdm.remove_redund(OFDM_RX_piloto, cpp, K, len(channelResponse) - 1, 'ZP-OLA')

        OFDM_demod_piloto = ofdm.DFT(OFDM_RX_ok_piloto)


        Hest_ls = (OFDM_demod_piloto / OFDM_data_piloto)

        H_LS = ofdm.block_real_imag(Hest_ls)
        scale = norm(H_LS)

        # Encontrando estimativa melhor do canal através da CE Net:
        H_hat = model2.predict(np.reshape(H_LS/scale, (1, 2 * K)))

        H_hat_ok = []
        for i in range(K):
            H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + K])
        H_hat_ok = np.asarray(H_hat_ok) * scale

        # Transmitindo agora os dados...

        oi = ofdm.IDFT(H_hat_ok)
        oi = np.reshape(oi, (K,))
        # h_est = channelResponse
        h_est = oi[0:len(channelResponse)]

        bits = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP = ofdm.SP(bits, K, mu)
        QAM = ofdm.Mapping(bits_SP, mapping_table)
        OFDM_data = ofdm.OFDM_symbol_all_sub(K, QAM)
        OFDM_time = ofdm.IDFT(OFDM_data)

        OFDM_TX = ofdm.add_redund(OFDM_time, cpp, K, 'ZP')

        if nonlinearity == True:
            # OFDM_TX = np.tanh(OFDM_TX)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX))
            CL = CR * sigma_x
            OFDM_TX = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX]

        bits_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_ant = ofdm.SP(bits_ant, K, mu)
        QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
        OFDM_data_ant = ofdm.OFDM_symbol_all_sub(K, QAM_ant)
        OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)

        OFDM_TX_ant = ofdm.add_redund(OFDM_time_ant, cpp, K, 'ZP')

        if nonlinearity == True:
            # OFDM_TX_ant = np.tanh(OFDM_TX_ant)
            CR = 1.3
            sigma_x = np.sqrt(np.var(OFDM_TX_ant))
            CL = CR * sigma_x
            OFDM_TX_ant = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX_ant]

        [OFDM_RX, sigma] = ofdm.channel(OFDM_TX, OFDM_TX_piloto, channelResponse, K + cpp, SNRdb)

        if cpp == 0:
            OFDM_RX_ok = OFDM_RX
        else:
            G_barra = ofdm.inversa_MMSE(h_est, K + cpp, cpp, SNRdb)
            L = len(channelResponse)-1
            OFDM_RX = OFDM_RX[(L - cpp):(K+cpp)]
            OFDM_RX_ok = np.dot(G_barra, OFDM_RX)

        equalized_Hest = ofdm.DFT(OFDM_RX_ok)

        y.append(ofdm.block_real_imag(equalized_Hest))
        y_true.append(ofdm.block_real_imag(OFDM_data))

    X_train = np.reshape(y, (size_data, 2 * K))
    Y_train = np.reshape(y_true, (size_data, 2 * K))

    return (X_train, Y_train)
#--------------------------------------------------------------------------------------------------------------
def data_set_min_red(y, y_true, channel_3g, size_data, model2, SNRdb, cpp, K, nonlinearity):
    for x in range(size_data):
        # Canal gerado:
        channelResponse = channel_3g[x]
        H_exact = np.fft.fft(channelResponse, K)

        # Gerando pilotos a serem transmitidos e recebidos:
        bits_piloto = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_piloto = ofdm.SP(bits_piloto, K, mu)
        QAM_piloto = ofdm.Mapping(bits_SP_piloto, mapping_table)
        OFDM_data_piloto = ofdm.OFDM_symbol_all_sub(K, QAM_piloto)
        
        bits_piloto_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_piloto_ant = ofdm.SP(bits_piloto_ant, K, mu)
        QAM_piloto_ant = ofdm.Mapping(bits_SP_piloto_ant, mapping_table)
        OFDM_data_piloto_ant = ofdm.OFDM_symbol_all_sub(K, QAM_piloto_ant)
       
        if sc_flag == True:
            OFDM_time_piloto = OFDM_data_piloto
            OFDM_time_piloto_ant = OFDM_data_piloto_ant
        else:
            OFDM_time_piloto = ofdm.IDFT(OFDM_data_piloto)
            OFDM_time_piloto_ant = ofdm.IDFT(OFDM_data_piloto_ant)
            
        OFDM_TX_piloto = ofdm.add_redund(OFDM_time_piloto, cpp, K, TX_type)
        OFDM_TX_piloto_ant = ofdm.add_redund(OFDM_time_piloto_ant, cpp, K, TX_type)

        if nonlinearity == True:
            # OFDM_TX_piloto = np.tanh(OFDM_TX_piloto)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX_piloto))
            CL = CR * sigma_x
            OFDM_TX_piloto = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX_piloto]
            sigma_x_ant = np.sqrt(np.var(OFDM_TX_piloto_ant))
            CL_ant = CR * sigma_x_ant
            OFDM_TX_piloto_ant = [i / abs(i) * CL_ant if abs(i) > CL_ant else i for i in OFDM_TX_piloto_ant]

        [OFDM_RX_piloto, sigma] = ofdm.channel(OFDM_TX_piloto, OFDM_TX_piloto_ant, channelResponse, K + cpp, SNRdb)

        OFDM_RX_ok_piloto = ofdm.remove_redund(OFDM_RX_piloto, cpp, K, len(channelResponse)-1, TX_type)
        
        OFDM_demod_piloto = ofdm.DFT(OFDM_RX_ok_piloto)
        OFDM_data_piloto = ofdm.DFT(OFDM_data_piloto)

        Hest_ls = (OFDM_demod_piloto / (OFDM_data_piloto+10**(-8)))

        H_LS = ofdm.block_real_imag(Hest_ls)
        scale = norm(H_LS)

        # Encontrando estimativa melhor do canal através da CE Net:
        H_hat = model2.predict(np.reshape(H_LS/scale, (1, 2 * K)))

        H_hat_ok = []
        for i in range(K):
            H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + K])
        H_hat_ok = np.asarray(H_hat_ok) * scale

        # Transmitindo agora os dados...
        bits = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP = ofdm.SP(bits, K, mu)
        QAM = ofdm.Mapping(bits_SP, mapping_table)
        OFDM_data = ofdm.OFDM_symbol_all_sub(K, QAM)
      
        if sc_flag == True:
            OFDM_time = OFDM_data
        else:
            OFDM_time = ofdm.IDFT(OFDM_data)

        OFDM_TX = ofdm.add_redund(OFDM_time, cpp, K, TX_type)

        if nonlinearity == True:
            # OFDM_TX = np.tanh(OFDM_TX)
            CR = 1.3
            # pdb.set_trace()
            sigma_x = np.sqrt(np.var(OFDM_TX))
            CL = CR * sigma_x
            OFDM_TX = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX]

        bits_ant = np.random.binomial(n=1, p=0.5, size=(K * mu))
        bits_SP_ant = ofdm.SP(bits_ant, K, mu)
        QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
        OFDM_data_ant = ofdm.OFDM_symbol_all_sub(K, QAM_ant)
        
        if sc_flag == True:
            OFDM_time_ant = OFDM_data_ant
        else:
            OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)

        OFDM_TX_ant = ofdm.add_redund(OFDM_time_ant, cpp, K, TX_type)

        if nonlinearity == True:
            # OFDM_TX_ant = np.tanh(OFDM_TX_ant)
            CR = 1.3
            sigma_x = np.sqrt(np.var(OFDM_TX_ant))
            CL = CR * sigma_x
            OFDM_TX_ant = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX_ant]

        [OFDM_RX, sigma] = ofdm.channel(OFDM_TX, OFDM_TX_piloto, channelResponse, K + cpp, SNRdb)

        OFDM_RX_ok = ofdm.remove_redund(OFDM_RX, cpp, K, len(channelResponse)-1, TX_type)

        OFDM_demod = ofdm.DFT(OFDM_RX_ok)

        H_hat_ok = np.asarray(H_hat_ok)

        equalized_Hest = ofdm.equalizer(OFDM_demod, H_hat_ok)

        if sc_flag == True:
            equalized_Hest = ofdm.IDFT(equalized_Hest)

        y.append(ofdm.block_real_imag(equalized_Hest))
        y_true.append(ofdm.block_real_imag(OFDM_data))

    X_train = np.reshape(y, (size_data, 2 * K))
    Y_train = np.reshape(y_true, (size_data, 2 * K))

    return (X_train, Y_train)
def Net1_wzp(SNRdb,cp,K):
    # Gerando data-set para a rede Net1
    (X_train,Y_train)=data_set_net1_zp([], [], channel_3g_train, 3000, SNRdb,cp,K, nonlinearity)
    # Gerando data-set treino
    (X_test,Y_test)=data_set_net1_zp([], [], channel_3g_test, 1000, SNRdb,cp,K, nonlinearity)

    X_train = preprocessing.normalize(X_train)
    Y_train = preprocessing.normalize(Y_train)
    X_test = preprocessing.normalize(X_test)
    Y_test = preprocessing.normalize(Y_test)

    #Constuindo Rede:
    model = Sequential()
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))
    model.add(Dense(2 * K, input_dim=2 * K))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=50, epochs=600, validation_data=(X_test, Y_test), verbose=1)

    return model,history

def Net1_wCPSC(SNRdb,cp,K):
    # Gerando data-set para a rede Net1
    (X_train,Y_train)=data_set_net1_CPSC([], [], channel_3g_train, 3000, SNRdb,cp,K, nonlinearity)
    # Gerando data-set treino
    (X_test,Y_test)=data_set_net1_CPSC([], [], channel_3g_test, 1000, SNRdb,cp,K, nonlinearity)

    X_train = preprocessing.normalize(X_train)
    Y_train = preprocessing.normalize(Y_train)
    X_test = preprocessing.normalize(X_test)
    Y_test = preprocessing.normalize(Y_test)

    #Constuindo Rede:
    model = Sequential()
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))
    model.add(Dense(2 * K, input_dim=2 * K))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=50, epochs=600, validation_data=(X_test, Y_test), verbose=1)

    return model,history

#--------------------------------------------------------------------------------------------------------------
def Net2_wzpzj(model_net1,SNRdb,cp,K):
    # Gerando data-set para a rede Net1
    (X_train,Y_train)=data_set_zpzj([], [], channel_3g_train, 15000,model_net1, SNRdb,cp,K, nonlinearity)
    # Gerando data-set treino
    (X_test,Y_test)=data_set_zpzj([], [], channel_3g_test, 1000,model_net1, SNRdb,cp,K, nonlinearity)

    X_train = preprocessing.normalize(X_train)
    Y_train = preprocessing.normalize(Y_train)
    X_test = preprocessing.normalize(X_test)
    Y_test = preprocessing.normalize(Y_test)


    #Constuindo Rede:
    model = Sequential()
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))

    model.add(Dense(2 * K, input_dim=2 * K))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=50, epochs=1500, validation_data=(X_test, Y_test), verbose=1)

    return model,history


def Net2_wcp(model_net1,SNRdb,cp,K):
    # Gerando data-set para a rede Net1
    (X_train,Y_train)=data_set_min_red([], [], channel_3g_train, 15000,model_net1, SNRdb,cp,K, nonlinearity)
    # Gerando data-set treino
    (X_test,Y_test)=data_set_min_red([], [], channel_3g_test, 1500,model_net1, SNRdb,cp,K, nonlinearity)

    # import pdb
    # pdb.set_trace()

    X_train = preprocessing.normalize(X_train)
    Y_train = preprocessing.normalize(Y_train)
    X_test = preprocessing.normalize(X_test)
    Y_test = preprocessing.normalize(Y_test)


    #Constuindo Rede:
    model = Sequential()
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))
    model.add(Dense(2 * K, activation='tanh', input_dim=2 * K))

    model.add(Dense(2 * K, input_dim=2 * K))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=50, epochs=1500, validation_data=(X_test, Y_test), verbose=1)

    return model,history


def CE_net_wcp(SNRdb,cp,K):
    # K = S-cp_met # CP pela metade, K = 64 - 9
    (X_train,Y_train,ww)=data_set_CE_net_min_red([], [], channel_3g_train, 3000, SNRdb,cp,K, nonlinearity)
    # Gerando data-set treino
    (X_test,Y_test,ww2)=data_set_CE_net_min_red([], [], channel_3g_test, 1000, SNRdb,cp,K, nonlinearity)

    #Pesos iniciais:
    bias = np.zeros(2*K)
    #Constuindo Rede:
    model=Sequential()
    model.add(Dense(2*K, input_dim=2*K, weights=[ww,bias]))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=50, epochs=600, validation_data = (X_test, Y_test), verbose=1)

    return model,history
#-----------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#[Net1_cp10_model,hist_net1_cp10] = Net1_wzp(SNRdb,CP,S-CP)
#Net1_cp10_model.save('models/SNR_'+str(SNRdb)+'_net1_cp10_model.h5')

#[Net1_wcp_model,hist_net1_wcp] = Net1_wzp(SNRdb,cp_met,S-cp_met)
#Net1_wcp_model.save('models/SNR_'+str(SNRdb)+'_net1_wcp_red_model.h5')

#[Net1_ncp_model,hist_net1_ncp] = Net1_wzp(SNRdb,0,S)
#Net1_ncp_model.save('models/SNR_'+str(SNRdb)+'_net1_ncp_model.h5')

#[Net2_ncp_model,hist_net2_ncp]= Net2_wcp(Net1_ncp_model,SNRdb,0,S)
#Net2_ncp_model.save('models/SNR_'+str(SNRdb)+'_net2_ncp_model.h5')

print ("SNRdb train é"+str(SNRdb))
print('Train CP ZERO')
[Net1_ncp_model,hist_net1_ncp] = Net1_wCPSC(SNRdb,0,S)
Net1_ncp_model.save(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net1_ncp_model.h5')
print('Train CP MET')
[Net1_wcp_model,hist_net1_wcp] = Net1_wCPSC(SNRdb,cp_met,S-cp_met)
Net1_wcp_model.save(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net1_wcp_red_model.h5')
print('Train CP FULL')
[Net1_cp10_model,hist_net1_cp10] = Net1_wCPSC(SNRdb,CP,S-CP)
Net1_cp10_model.save(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net1_cp10_model.h5')
print('Train NET MR')
[Net2_ncp_model,hist_net2_ncp]= Net2_wcp(Net1_ncp_model,SNRdb,0,S)
Net2_ncp_model.save(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net2_ncp_model.h5')

#######################################################################################################
#[Net1_wzp_model,hist_net1_wzp] = Net1_wzp(SNRdb,cp_met,S-cp_met)
#Net1_wzp_model.save('models/SNR_'+str(SNRdb)+'_net1_wzpzj_red_model.h5')

#[Net1_zp10_model,hist_net1_zp10] = Net1_wzp(SNRdb,CP,S-CP)
#Net1_zp10_model.save('models/SNR_'+str(SNRdb)+'_net1_zpzj10_model.h5')
#
fig = plt.figure(figsize=(6, 6))
plt.plot(hist_net1_cp10.history['loss'],label='CP = L(train)')
plt.plot(hist_net1_cp10.history['val_loss'],label='CP = L(validation)')
plt.plot(hist_net1_wcp.history['loss'],label='CP = L/2 (train)')
plt.plot(hist_net1_wcp.history['val_loss'],label='CP = L/2 (validation)')
plt.plot(hist_net1_ncp.history['loss'],label='CP = 0 (train)')
plt.plot(hist_net1_ncp.history['val_loss'],label='CP = 0 (validation)')
plt.plot(hist_net2_ncp.history['loss'],label='MR net CP = 0 (train)')
plt.plot(hist_net2_ncp.history['val_loss'],label='MR net CP = 0 (validation)')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(True, which='minor',linestyle=':')
plt.grid(True, which='major')
plt.legend(framealpha=1, frameon=True)
#plt.show()
fig.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_1.png')
fig.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_1.pdf')

fig1 = plt.figure(figsize=(6, 6))
plt.plot(hist_net1_cp10.history['loss'],label='CP = L(train)')
plt.plot(hist_net1_cp10.history['val_loss'],label='CP = L(validation)')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(True, which='minor',linestyle=':')
plt.grid(True, which='major')
plt.legend(framealpha=1, frameon=True)
#plt.show()
fig1.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_CP_L.png')
fig1.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_CP_L.pdf')

fig2 = plt.figure(figsize=(6, 6))
plt.plot(hist_net1_wcp.history['loss'],label='CP = L/2 (train)')
plt.plot(hist_net1_wcp.history['val_loss'],label='CP = L/2 (validation)')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(True, which='minor',linestyle=':')
plt.grid(True, which='major')
plt.legend(framealpha=1, frameon=True)
#plt.show()
fig2.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_CP_met.png')
fig2.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_CP_met.pdf')

fig3 = plt.figure(figsize=(6, 6))
plt.plot(hist_net1_ncp.history['loss'],label='CP = 0 (train)')
plt.plot(hist_net1_ncp.history['val_loss'],label='CP = 0 (validation)')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(True, which='minor',linestyle=':')
plt.grid(True, which='major')
plt.legend(framealpha=1, frameon=True)
#plt.show()
fig3.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_CP_zero.png')
fig3.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_CP_zero.pdf')

fig4 = plt.figure(figsize=(6, 6))
plt.plot(hist_net2_ncp.history['loss'],label='MR net CP = 0 (train)')
plt.plot(hist_net2_ncp.history['val_loss'],label='MR net CP = 0 (validation)')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid(True, which='minor',linestyle=':')
plt.grid(True, which='major')
plt.legend(framealpha=1, frameon=True)
#plt.show()
fig4.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_MRnet_CP_zero.png')
fig4.savefig(str_linear+channel_type+'/figuras/'+str(QAM)+'QAM/'+str(S)+'SC''/SNRdb_'+str(SNRdb)+'_train_MRnet_CP_zero.pdf')

#
#
# fig = plt.figure(figsize=(6, 6))
# plt.plot(hist_net1_wzp.history['loss'],label='Net1 (train)')
# plt.plot(hist_net1_wzp.history['val_loss'],label='Net1 (validation)')
# plt.plot(hist_net2_wzp.history['loss'],label='net2 (train)')
# plt.plot(hist_net2_wzp.history['val_loss'],label='net2 (validation)')
# plt.xscale('linear')
# plt.yscale('log')
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# plt.grid(True, which='minor',linestyle=':')
# plt.grid(True, which='major')
# plt.legend(framealpha=1, frameon=True)
# plt.show()
# fig.savefig('train_1.png')
# fig.savefig('train_1.pdf')

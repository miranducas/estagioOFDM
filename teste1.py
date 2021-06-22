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


S = 64                         # Number of OFDM sub-carriers
SNRdb=0                    # Signal-to-noise ratio (value)
ensemble = 100                     # Number of independent runs to average    
mu = 4                          # Number of bits per symbol
train = False                     # True if training will be performed
nonlinearity = False            # True if nonlinearity is added at TX
channel_type = 'PED'            # 'PED' or 'VEH'
TX_type = 'CP-SC'
#-------------------------------------------------------------------------------------------------------------------------------
if channel_type=='PED':
    Vchannel = spio.loadmat('channel_ped.mat')
    channel_3g_train = Vchannel['y_ped_train_all']
    channel_3g_test = Vchannel['y_ped_test']
if channel_type=='VEH':
    Vchannel = spio.loadmat('channel_veh.mat')
    channel_3g_train = Vchannel['y_veh_train']
    channel_3g_test = Vchannel['y_veh_test']

taps = channel_3g_test.shape[1]
CP = taps-1
cp_met = CP//2

[mapping_table,demapping_table] = ofdm.QAM_mod(2**mu)


def data_set_net1_CPSC(H_LS, H_true, channel_3g, size_data, SNRdb, cpp, K, nonlinearity):    
    for x in range(size_data):
        print(str(x))
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
            
        if TX_type == 'CP-SC':
            #print('TX EH CP-SC iteracao '+str(x))
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
        #print('RX OK EH'+str(OFDM_RX_ok))
        #print('DATA  EH'+str(OFDM_data))

        OFDM_demod = ofdm.DFT(OFDM_RX_ok)
        OFDM_data2 = ofdm.DFT(OFDM_data)

        #OFDM_demod = OFDM_RX_ok
        #Hest_ls  = (OFDM_demod_piloto2/OFDM_data_piloto)
        H_aux =(OFDM_demod/OFDM_data2)
        #H_aux = ofdm.DFT(H_aux)
        H_LS.append(ofdm.block_real_imag(H_aux))
        H_true.append(ofdm.block_real_imag(H_exact))
    X_train = np.asarray(H_LS)
    Y_train = np.asarray(H_true)
    return (X_train, Y_train)

(X_train2,Y_train2)=data_set_net1_CPSC([], [], channel_3g_train, 3, SNRdb,CP,S-CP, nonlinearity)


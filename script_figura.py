#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pathlib as pb
import os
#import plotly.graph_objects as go
import numpy as np
import fig_lib
import pandas as pd
import scipy.io as spio
################################################################################################################################
# This script is used to generate the BER curve by using the simulation results in /results_ber folder.
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------#
channel_type = 'PED'
SC=64
QAM = 16
ensemble = 2000
nonlinearity = False

#Estrutura condicional que define se o modelo é para veículos ou para pedestres 
if channel_type=='PED':
    Vchannel = spio.loadmat('channel_ped.mat')
    #channel_3g_train = Vchannel['y_ped_train']
    channel_3g_test = Vchannel['y_ped_test']
if channel_type=='VEH':
    Vchannel = spio.loadmat('channel_veh.mat')
    channel_3g_train = Vchannel['y_veh_train']
    channel_3g_test = Vchannel['y_veh_test']
if channel_type =='VEHPED':
    Vchannel = spio.loadmat('channel_vehped.mat')
    channel_3g_train = Vchannel['y_vehped_train']
    channel_3g_test = Vchannel['y_vehped_test']

taps = channel_3g_test.shape[1]
CP = taps-1

if nonlinearity == True:
    str_linear = 'nao_linear/'
    str_linear2 = 'nao_linear'
if nonlinearity == False:
    str_linear = 'linear/'
    str_linear2 = 'linear'
df=pd.DataFrame({'x': range(1,101), 'y': np.random.randn(100)*15+range(1,101), 'z': (np.random.randn(100)*15+range(1,101))*2 })




# Going to results folder and selecting ber results files:
#[SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10, ber_net1_wzp, ber_wzpzj, ber_net1_zp10] = fig_lib.BER_SNR('results_ber')#------------------------------------------------------------------------------------------------------------------------------#
#[SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10] = fig_lib.BER_SNR('results_ber')
[SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10] = fig_lib.BER_SNR(str_linear+channel_type+'/results_ber/'+str(QAM)+'QAM/'+str(SC)+'SC')
#[SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact] = fig_lib.BER_SNR('results_ber')

# Visualizing bit-error-rate vs SNR:

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(15, 7))
plt.plot(SNRdb_range, ber_ls, marker='x', color='red',label='CP-SC: LS+BFDE (K=0)')
plt.plot(SNRdb_range, ber_zf_mr, marker='d', color='gold',label='CP-SC: LS+BFDE (K=5)')
plt.plot(SNRdb_range, ber_wcp_exact, marker='o', color='green',label='CP-SC: Exact+BFDE (K=10)')
plt.plot(SNRdb_range, ber_wcp, marker='s', color='deepskyblue',label='CP-SC: LMMSE+BFDE (K=10)')
plt.plot(SNRdb_range,  ber_net1_wcp,  marker='^', color='black',label='CP-SC: ICE+BFDE (K=5)')
plt.plot(SNRdb_range,  ber_net1_ncp, marker='^', color='purple',label='CP-SC: ICE+BFDE (K=0)')
plt.plot(SNRdb_range,  ber_net2_ncp, marker='v', color='hotpink',label='CP-SC: ICE+SDMR (K=0)')
plt.plot(SNRdb_range,  ber_net1_cp10, marker='^', color='darkorange',label='CP-SC: ICE+BFDE (K=10)')

#plt.plot(SNRdb_range,  ber_net1_wzp,  marker='*',  linestyle='dashed', color='crimson',label='ZPZJ: ICE+reg (K=5)')
#plt.plot(SNRdb_range,  ber_wzpzj,  marker='*',  linestyle='dashed',color='blue',label='ZPZJ: LS+reg (K=5)')
#plt.plot(SNRdb_range,  ber_net1_zp10,  marker='*',  linestyle='dashed', color='darkred',label='ZPZJ: ICE+MMSE (K=10)')

# plt.legend(framealpha=1, frameon=True, ncol =2);
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.grid(True, which='minor',linestyle=':')
plt.grid(True, which='major')
plt.ylim((10**(-6), 1))
plt.subplots_adjust(right=0.7)
plt.show()


output_filename = 'ber_'+str(QAM)+'QAM_'+str(ensemble)+'main_'+str_linear2+'_'+str(SC)+'SC'
#output_filename = 'ber'
fig.savefig(str(SC)+'SC/'+output_filename+'.png')
fig.savefig(str(SC)+'SC/'+output_filename+'.pdf')
################################################################################################################################




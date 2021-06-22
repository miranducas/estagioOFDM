#!/usr/bin/env python
# coding: utf-8
from typing import List, Any

import statistics as sta
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import ofdm
import ofdm_methods
import scipy.io as spio
################################################################################################################################
#-------------------------------------------------------------------------------------------------------------------------------
# System Parameters Decription
#-------------------------------------------------------------------------------------------------------------------------------
S = 64                           # Number of OFDM sub-carriers
SNRdb=40                    # Signal-to-noise ratio (value)
ensemble = 2000                     # Number of independent runs to average
mu = 4                          # Number of bits per symbol
QAM = 2**mu
nonlinearity = False          # True if nonlinearity is added at TX
channel_type = 'PED'            # 'PED' or 'VEH' or 'VEHPED'
TX_type = 'CP-SC'
sc_flag = True
#-------------------------------------------------------------------------------------------------------------------------------
if channel_type=='PED':
    Vchannel = spio.loadmat('channel_ped.mat')
    channel_3g = Vchannel['y_ped_main']
if channel_type=='VEH':
    Vchannel = spio.loadmat('channel_veh.mat')
    channel_3g = Vchannel['y_veh_main']
if channel_type =='VEHPED':
    Vchannel = spio.loadmat('channel_vehped.mat')
    channel_3g = Vchannel['y_vehped_main']

taps = channel_3g.shape[1]
CP = taps-1
cp_met = CP//2

if nonlinearity == True:
    str_linear = 'nao_linear/'
if nonlinearity == False:
    str_linear = 'linear/'

[mapping_table,demapping_table] = ofdm.QAM_mod(2**mu)


Net1_wcp_model = tf.keras.models.load_model(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net1_wcp_red_model.h5')
Net1_ncp_model = tf.keras.models.load_model(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net1_ncp_model.h5')
Net2_ncp_model = tf.keras.models.load_model(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net2_ncp_model.h5')
Net1_cp10_model = tf.keras.models.load_model(str_linear+channel_type+'/models/'+str(QAM)+'QAM/'+str(S)+'SC''/SNR_'+str(SNRdb)+'_net1_cp10_model.h5')



#Net1_wzp_model = tf.keras.models.load_model('models/SNR_'+str(SNRdb)+'_net1_wzpzj_red_model.h5')
#Net1_zp10_model = tf.keras.models.load_model('models/SNR_'+str(SNRdb)+'_net1_zpzj10_model.h5')
#--------------------------------------------------------------------------------------------------------------
# MAIN
#--------------------------------------------------------------------------------------------------------------
ber_min_red_aux=[]
ber_wcp_aux=[]
ber_wcp_exact_aux=[]
ber_cenet_new_aux=[]
ber_cenet_aux=[]
ber_ls_aux=[]
ber_zf_mr_aux=[]
ber_true_aux=[]
ber_cenet_min_red_aux=[]
ber_net1_wcp_aux=[]
ber_net1_ncp_aux=[]
ber_net2_ncp_aux=[]
ber_net1_cp3_aux=[]
ber_net2_cp3_aux=[]
ber_net1_cp10_aux=[]
#ber_net1_wzp_aux=[]
#ber_wzpzj_aux=[]
#ber_net1_zp10_aux=[]

for ee in range(ensemble):
    # Canal gerado:
    channelResponse =  channel_3g[ee]
    H_exact = np.fft.fft(channelResponse,(S-CP)) # H exato quando CP=L
    H_true=(ofdm.block_real_imag(H_exact))   
#------------------------------------------------------------------------------------------------------------------------------#
#       Fase dos pilotos
#-------------------------------------------------------------------------------------------------------------------------------
    #Gerando pilotos a serem transmitidos e recebidos:
    #Piloto geral:
    bits_piloto_all = np.random.binomial(n=1, p=0.5, size=(S*mu))

#------------------------------------------------------------------------------------------------------------------------------#
#       Fase dos dados
#-------------------------------------------------------------------------------------------------------------------------------
    #Transmitindo agora os dados...
    bits_all = np.random.binomial(n=1, p=0.5, size=(S * mu))

    QAM_est_ls = ofdm_methods.LS(bits_piloto_all, bits_all, S, 0, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_zf_mr = ofdm_methods.LS(bits_piloto_all, bits_all, S-cp_met, cp_met, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_wcp = ofdm_methods.LMMSE(bits_piloto_all, bits_all, S-CP, CP, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_wcp_exact = ofdm_methods.Exact(bits_piloto_all, bits_all, S-CP, CP, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_net1_wcp = ofdm_methods.Net1(Net1_wcp_model ,bits_piloto_all, bits_all, S-cp_met, cp_met, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_net1_ncp = ofdm_methods.Net1(Net1_ncp_model ,bits_piloto_all, bits_all, S, 0, mu, mapping_table, demapping_table, SNRdb,  nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_net2_ncp = ofdm_methods.MR(Net1_ncp_model,Net2_ncp_model, bits_piloto_all, bits_all, S, 0, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type)
    QAM_est_net1_cp10 = ofdm_methods.Net1(Net1_cp10_model, bits_piloto_all, bits_all, S-CP, CP, mu, mapping_table, demapping_table, SNRdb,  nonlinearity, channelResponse,sc_flag,TX_type)

    #QAM_est_wzpzj = ofdm_methods.ZPZJ(bits_piloto_all, bits_all, S - cp_met, cp_met, mu, mapping_table, demapping_table, SNRdb, nonlinearity, channelResponse)
    #QAM_est_net1_wzp = ofdm_methods.ZPZJ_1(Net1_wzp_model ,bits_piloto_all, bits_all, S-cp_met, cp_met, mu, mapping_table, demapping_table, SNRdb,  nonlinearity, channelResponse)
    #QAM_est_net1_zp10 = ofdm_methods.ZPZJ_1(Net1_zp10_model ,bits_piloto_all, bits_all, S-CP, CP, mu, mapping_table, demapping_table, SNRdb,  nonlinearity, channelResponse)


    ber_ls_aux.append(QAM_est_ls)
    ber_zf_mr_aux.append(QAM_est_zf_mr)
    ber_wcp_aux.append(QAM_est_wcp)
    ber_wcp_exact_aux.append(QAM_est_wcp_exact)

    ber_net1_wcp_aux.append(QAM_est_net1_wcp)
    ber_net1_ncp_aux.append(QAM_est_net1_ncp)
    ber_net2_ncp_aux.append(QAM_est_net2_ncp)
    ber_net1_cp10_aux.append(QAM_est_net1_cp10)
    
    #ber_net1_wzp_aux.append(QAM_est_net1_wzp)
    #ber_wzpzj_aux.append(QAM_est_wzpzj)
    #ber_net1_zp10_aux.append(QAM_est_net1_zp10)
    print(ee)

ber_ls=sta.mean(ber_ls_aux)
ber_zf_mr=sta.mean(ber_zf_mr_aux)
ber_wcp=sta.mean(ber_wcp_aux)
ber_wcp_exact=sta.mean(ber_wcp_exact_aux)
ber_net1_wcp = sta.mean(ber_net1_wcp_aux)
ber_net1_ncp = sta.mean(ber_net1_ncp_aux)
ber_net2_ncp = sta.mean(ber_net2_ncp_aux)
ber_net1_cp10 = sta.mean(ber_net1_cp10_aux)

#ber_net1_wzp = sta.mean(ber_net1_wzp_aux)
#ber_wzpzj = sta.mean(ber_wzpzj_aux)
#ber_net1_zp10 = sta.mean(ber_net1_zp10_aux)

file_name = str_linear+channel_type+'/results_ber/'+str(QAM)+'QAM/'+str(S)+'SC''/BER_'+str(ensemble)+'main_'+str(QAM)+'qam'+'_cp_'+str(CP)+'_snr_'+str(SNRdb)+'.pickle'

with open(file_name, 'wb') as f:
    #pickle.dump([ ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact], f)
    pickle.dump([ ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10], f)
    #pickle.dump([ ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10, ber_net1_wzp, ber_wzpzj, ber_net1_zp10], f)

print('salvo')
print ("SNRdb main é"+str(SNRdb))




#----------------------------------------------------------------------------------------------------------------
# BER
#----------------------------------------------------------------------------------------------------------------
#
# fig = plt.figure(figsize=(8, 7))
# plt.plot(SNRdb, ber_ls, marker='x', color='red',label='LS+BFDE (K=0)')
# plt.plot(SNRdb, ber_zf_mr, marker='d', color='gold',label='LS+BFDE (K=5)')
# plt.plot(SNRdb, ber_wcp_exact, marker='o', color='green',label='Exact+BFDE (K=10)')
# plt.plot(SNRdb, ber_wcp, marker='s', color='deepskyblue',label='LMMSE+BFDE (K=10)')
# plt.plot(SNRdb,  ber_net1_wzp,  marker='^', color='lightcoral',label='zpzj CE+E_LMMSE (K=5)')
# plt.plot(SNRdb,  ber_wzpzj,  marker='^', color='silver',label='zpzj LS+E_LMMSE (K=5)')
# plt.plot(SNRdb,  ber_net1_wcp,  marker='^', color='black',label='CE+BFDE (K=5)')
# plt.plot(SNRdb,  ber_net1_ncp, marker='^', color='purple',label='CE+BFDE (K=0)')
# plt.plot(SNRdb,  ber_net2_ncp, marker='v', color='hotpink',label='CE+SDMR (K=0)')
# plt.plot(SNRdb,  ber_net1_cp10, marker='^', color='darkorange',label='CE+BFDE (K=10)')
# plt.plot(SNRdb,  ber_net1_zp10,  marker='^', color='darkred',label='zpzj CE+E_ZF (K=10)')
# plt.legend(framealpha=1, frameon=True, ncol =2);
# plt.xscale('linear')
# plt.yscale('log')
# plt.xlabel('SNR(dB)')
# plt.ylabel('BER')
# plt.grid(True, which='minor',linestyle=':')
# plt.grid(True, which='major')
# plt.ylim((10**(-6), 1))
#
# plt.show()
#
# output_filename = 'ber'
# fig.savefig(output_filename+'.png')
# fig.savefig(output_filename+'.pdf')

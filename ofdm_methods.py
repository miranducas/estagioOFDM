import numpy as np
import ofdm
import ofdmclass
import ofdm_func
from sklearn import preprocessing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pdb
from cmath import sqrt
from scipy.linalg import norm
import matplotlib.pyplot as plt



def ZPZJ(bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb, nonlinearity, channelResponse):

    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_RX, OFDM_data_piloto]= ofdm_func.symb_transmission2(pilot_symb, channelResponse, True, nonlinearity)

    OFDM_demod_piloto = ofdm.DFT(OFDM_RX)
    Hest_ls  = (OFDM_demod_piloto/OFDM_data_piloto),
    oi = ofdm.IDFT(Hest_ls)
    oi = np.reshape(oi, (N,))
    # h_est = channelResponse
    h_est = oi[0:len(channelResponse)]


    G_barra = ofdm.inversa_MMSE(h_est, N+K, K, SNRdb)

    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_RX, OFDM_data_symb] = ofdm_func.symb_transmission2(data_symb, channelResponse,False, nonlinearity)

    OFDM_RX_ok = np.dot(G_barra, OFDM_RX)
    equalized_Hest_ls = ofdm.DFT(OFDM_RX_ok)

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est

def ZPZJ_1(model1,bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb, nonlinearity, channelResponse):

    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_RX, OFDM_data_piloto]= ofdm_func.symb_transmission2(pilot_symb, channelResponse, True, nonlinearity)

    OFDM_demod_piloto = ofdm.DFT(OFDM_RX)
    Hest_ls  = (OFDM_demod_piloto/OFDM_data_piloto)

    H_LS = ofdm.block_real_imag(Hest_ls)
    scale = norm(H_LS)
    # Encontrando estimativa melhor do canal através da CE Net:
    H_hat = model1.predict(np.reshape(H_LS/scale, (1, 2 * N)))
    H_hat_ok = []
    for i in range(N):
        H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + N])
    H_hat_ok = np.asarray(H_hat_ok)*scale

    oi = ofdm.IDFT(H_hat_ok)
    oi = np.reshape(oi, (N,))
    # h_est = channelResponse
    h_est = oi[0:len(channelResponse)]


    G_barra = ofdm.inversa_MMSE(h_est, N+K, K, SNRdb)

    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb )
    [OFDM_RX, OFDM_data_symb] = ofdm_func.symb_transmission2(data_symb, channelResponse,False, nonlinearity)

    OFDM_RX_ok = np.dot(G_barra, OFDM_RX)
    equalized_Hest_ls = ofdm.DFT(OFDM_RX_ok)

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est



def ZPZJ_2(model1, model, bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb, nonlinearity, channelResponse):

    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_RX, OFDM_data_piloto]= ofdm_func.symb_transmission2(pilot_symb, channelResponse, True, nonlinearity)

    OFDM_demod_piloto = ofdm.DFT(OFDM_RX)
    Hest_ls  = (OFDM_demod_piloto/OFDM_data_piloto)

    H_LS = ofdm.block_real_imag(Hest_ls)
    scale = norm(H_LS)
    # Encontrando estimativa melhor do canal através da CE Net:
    H_hat = model1.predict(np.reshape(H_LS / scale, (1, 2 * N)))
    H_hat_ok = []
    for i in range(N):
        H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + N])
    H_hat_ok = np.asarray(H_hat_ok) * scale

    oi = ofdm.IDFT(H_hat_ok)
    oi = np.reshape(oi, (N,))
    # h_est = channelResponse
    h_est = oi[0:len(channelResponse)]


    G_barra = ofdm.inversa_MMSE(h_est, N+K, K, SNRdb)

    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_RX, OFDM_data_symb] = ofdm_func.symb_transmission2(data_symb, channelResponse,False, nonlinearity)

    OFDM_RX_ok = np.dot(G_barra, OFDM_RX)
    equalized_Hest_ls = ofdm.DFT(OFDM_RX_ok)

    y = (ofdm.block_real_imag(equalized_Hest_ls))
    oi2 = np.reshape(y, (1, -1))
    mrscaler = norm(oi2)
    y_hat = model.predict(np.reshape(oi2 / mrscaler, (1, 2 * N)))

    u_hat = []
    for i in range(N):
        u_hat.append(y_hat[0, i] + 1j * y_hat[0, i + N])
    # print(u_hat)
    # pdb.set_trace()
    u_hat = np.asarray(u_hat) * mrscaler

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,u_hat,bits_all)
    return QAM_est


def MR(model1,model,bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb , nonlinearity, channelResponse,sc_flag,TX_type):
    #print('MR CP='+str(K)+' e SNR='+str(SNRdb))
    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod_piloto, OFDM_data_piloto, OFDM_TX_piloto]= ofdm_func.symb_transmission(pilot_symb, channelResponse, True, nonlinearity,[],sc_flag,TX_type)
    if  sc_flag == True:
        OFDM_data_piloto = ofdm.DFT(OFDM_data_piloto) 
    Hest_ls  = (OFDM_demod_piloto/(OFDM_data_piloto+10**(-8)))

    H_LS = ofdm.block_real_imag(Hest_ls)
    scale = norm(H_LS)
    # Encontrando estimativa melhor do canal através da CE Net:
    H_hat = model1.predict(np.reshape(H_LS / scale, (1, 2 * N)))
    H_hat_ok = []
    for i in range(N):
        H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + N])
    H_hat_ok = np.asarray(H_hat_ok) * scale

    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod, OFDM_data_symb,ant] = ofdm_func.symb_transmission(data_symb, channelResponse,False, nonlinearity,OFDM_TX_piloto,sc_flag,TX_type)
    
    if  sc_flag == True:
        #OFDM_demod = ofdm.IDFT(OFDM_demod)
        equalized_Hest_cenet = ofdm.equalizer(OFDM_demod, H_hat_ok)
        equalized_Hest_cenet = ofdm.IDFT(equalized_Hest_cenet)
    else:
        equalized_Hest_cenet = ofdm.equalizer(OFDM_demod, H_hat_ok)

    y = (ofdm.block_real_imag(equalized_Hest_cenet))
    oi2 = np.reshape(y, (1, -1))
    mrscaler = norm(oi2)
    y_hat = model.predict(np.reshape(oi2/mrscaler, (1, 2 * N)))


    u_hat = []
    for i in range(N):
        u_hat.append(y_hat[0, i] + 1j * y_hat[0, i + N])
    # print(u_hat)
    # pdb.set_trace()
    u_hat = np.asarray(u_hat)*mrscaler

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,u_hat,bits_all)
    return QAM_est

def Net1(model1,bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type):
    #print('NET1 CP='+str(K)+' e SNR='+str(SNRdb))
    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod_piloto, OFDM_data_piloto, OFDM_TX_piloto]= ofdm_func.symb_transmission(pilot_symb, channelResponse, True, nonlinearity,[],sc_flag,TX_type)
   
    if  sc_flag == True:
        OFDM_data_piloto = ofdm.DFT(OFDM_data_piloto) 
    Hest_ls  = (OFDM_demod_piloto/(OFDM_data_piloto+10**(-8)))
    
    H_LS = ofdm.block_real_imag(Hest_ls)
    scale=norm(H_LS)
    # Encontrando estimativa melhor do canal através da CE Net:
    H_hat = model1.predict(np.reshape(H_LS/scale, (1, 2 * N)))
    H_hat_ok = []
    for i in range(N):
        H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + N])
    H_hat_ok = np.asarray(H_hat_ok)*scale

    # plt.plot(np.fft.fft(channelResponse, N), '-bo')
    # plt.plot(Hest_ls, '-r+')
    # plt.plot(H_hat_ok, '-mx')
    # import pdb
    # pdb.set_trace()
    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb )
    [OFDM_demod, OFDM_data_symb,ant] = ofdm_func.symb_transmission(data_symb, channelResponse,False, nonlinearity,OFDM_TX_piloto,sc_flag,TX_type)
   
    if  sc_flag == True:
       # OFDM_demod = ofdm.IDFT(OFDM_demod)
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, H_hat_ok)
        equalized_Hest_ls = ofdm.IDFT(equalized_Hest_ls)
    else:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, H_hat_ok)
    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est

def CE(model1,bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb , nonlinearity, channelResponse,sc_flag,TX_type):

    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod_piloto, OFDM_data_piloto, OFDM_TX_piloto]= ofdm_func.symb_transmission(pilot_symb, channelResponse, True, nonlinearity,[],sc_flag,TX_type)
   
    if  sc_flag == True:
        OFDM_data_piloto = ofdm.DFT(OFDM_data_piloto) 

    Hest_ls  = (OFDM_demod_piloto/OFDM_data_piloto)
    H_LS = ofdm.block_real_imag(Hest_ls)
    # Encontrando estimativa melhor do canal através da CE Net:
    H_hat = model1.predict(np.reshape(H_LS, (1, 2 * N)))
    H_hat_ok = []
    for i in range(N):
        H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + N])
    H_hat_ok = np.asarray(H_hat_ok)

    # plt.plot(np.fft.fft(channelResponse, N), '-bo')
    # plt.plot(Hest_ls, '-r+')
    # plt.plot(H_hat_ok, '-mx')
    # import pdb
    # pdb.set_trace()
    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod, OFDM_data_symb,ant] = ofdm_func.symb_transmission(data_symb, channelResponse,False, nonlinearity,OFDM_TX_piloto,sc_flag,TX_type)
    equalized_Hest_ls = ofdm.equalizer(OFDM_demod, H_hat_ok)

    if  sc_flag == True:
        #OFDM_demod = ofdm.IDFT(OFDM_demod)
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, H_hat_ok)
        equalized_Hest_ls = ofdm.IDFT(equalized_Hest_ls)

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est

def LS(bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb, nonlinearity, channelResponse,sc_flag,TX_type):
    #print('LS CP='+str(K)+' e SNR='+str(SNRdb))
    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod_piloto, OFDM_data_piloto, OFDM_TX_piloto]= ofdm_func.symb_transmission(pilot_symb, channelResponse, True, nonlinearity,[],sc_flag,TX_type)

    if  sc_flag == True:
        OFDM_data_piloto = ofdm.DFT(OFDM_data_piloto)
        #OFDM_demod_piloto = ofdm.IDFT(OFDM_demod_piloto)
    Hest_ls  = (OFDM_demod_piloto/(OFDM_data_piloto+10**(-8)))
    #Hest_ls  = (OFDM_demod_piloto/OFDM_data_piloto)
    #Hest_ls = ofdm.DFT(Hest_ls)
    #print('1-tamanho é'+str(Hest_ls.shape))
    #print('agora é tamanho é'+str(OFDM_TX_piloto.shape))   
    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb)
    [OFDM_demod, OFDM_data_symb,ant] = ofdm_func.symb_transmission(data_symb, channelResponse,False, nonlinearity,OFDM_TX_piloto,sc_flag,TX_type)
    #print('2-tamanho dessa variavel é'+str(OFDM_demod.shape))
    if  sc_flag == True:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, Hest_ls)
        equalized_Hest_ls = ofdm.IDFT(equalized_Hest_ls)
    else:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, Hest_ls)

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est

def LMMSE(bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb , nonlinearity, channelResponse,sc_flag,TX_type):
    #print('MMSE SNR='+str(SNRdb))
    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N,K, mu, mapping_table, SNRdb)
    [OFDM_demod_piloto, OFDM_data_piloto, OFDM_TX_piloto]= ofdm_func.symb_transmission(pilot_symb, channelResponse, True, nonlinearity,[],sc_flag,TX_type)
    
    if  sc_flag == True:
        OFDM_data_piloto = ofdm.DFT(OFDM_data_piloto)
        #OFDM_demod_piloto = ofdm.IDFT(OFDM_demod_piloto)
    Hest_wcp = ofdm.MMSEchannelEstimate(OFDM_demod_piloto,OFDM_data_piloto+10**(-8),N,N,channelResponse,SNRdb)
    #Hest_wcp = ofdm.MMSEchannelEstimate(OFDM_demod_piloto,OFDM_data_piloto+10**(-8),N,N,channelResponse,SNRdb)
    #Hest_wcp = ofdm.DFT(Hest_wcp)
    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb )
    [OFDM_demod, OFDM_data_symb,ant] = ofdm_func.symb_transmission(data_symb, channelResponse,False, nonlinearity,OFDM_TX_piloto,sc_flag,TX_type)
    if  sc_flag == True:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, Hest_wcp)
        equalized_Hest_ls = ofdm.IDFT(equalized_Hest_ls)
    else:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, Hest_wcp)    
    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est

def Exact(bits_piloto_all, bits_all, N, K, mu, mapping_table,demapping_table, SNRdb , nonlinearity, channelResponse,sc_flag,TX_type):
    #print('Exact SNR='+str(SNRdb))
    Hest_exact = np.fft.fft(channelResponse, N)  # H exato quando CP=L
    pilot_symb = ofdmclass.symbol_block(bits_piloto_all, N,K, mu, mapping_table, SNRdb)
    [OFDM_demod_piloto, OFDM_data_piloto, OFDM_TX_piloto]= ofdm_func.symb_transmission(pilot_symb, channelResponse, True, nonlinearity,[],sc_flag,TX_type)
   
    data_symb = ofdmclass.symbol_block(bits_all, N, K, mu, mapping_table, SNRdb )
    [OFDM_demod, OFDM_data_symb,ant] = ofdm_func.symb_transmission(data_symb, channelResponse,False, nonlinearity,OFDM_TX_piloto,sc_flag,TX_type)
 
    if  sc_flag == True:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, Hest_exact)
        equalized_Hest_ls = ofdm.IDFT(equalized_Hest_ls)
    else:
        equalized_Hest_ls = ofdm.equalizer(OFDM_demod, Hest_exact)

    QAM_est = ofdm_func.symb_ber(data_symb, demapping_table,equalized_Hest_ls,bits_all)
    return QAM_est

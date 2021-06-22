
import numpy as np
import ofdm


def symb_transmission(symbol_block, channelResponse, pilot, nonlinearity, OFDM_TX_piloto,sc_flag,TX_type):
    S = symbol_block.S
    N = symbol_block.N
    K = symbol_block.K
    mu = symbol_block.mu
    mapping_table = symbol_block.mapping_table
    bits_all = symbol_block.bits
    SNRdb = symbol_block.SNRdb
    L = len(channelResponse)-1
    
    bits = bits_all[0:(N*mu)]
    bits_SP = ofdm.SP(bits,N,mu)
    QAM = ofdm.Mapping(bits_SP,mapping_table)
    OFDM_data = ofdm.OFDM_symbol_all_sub(N,QAM)
    

    bits_ant = np.random.binomial(n=1, p=0.5, size=(N * mu))
    bits_SP_ant = ofdm.SP(bits_ant, N, mu)
    QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
    OFDM_data_ant = ofdm.OFDM_symbol_all_sub(N, QAM_ant)
   

    if sc_flag == True:
       OFDM_time = OFDM_data
       OFDM_time_ant = OFDM_data_ant       
    else:
        OFDM_time = ofdm.IDFT(OFDM_data)
        OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)

    OFDM_TX = ofdm.add_redund(OFDM_time, K, N, TX_type)
    OFDM_TX_ant =ofdm.add_redund(OFDM_time_ant, K, N, TX_type)

      
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

    #if TX_type == 'CP-SC':
    #    channel = ofdm.DFT(channelResponse)    
    #else:
    channel = channelResponse
    if pilot == True:
        [OFDM_RX,sigma] = ofdm.channel(OFDM_TX,OFDM_TX_ant,channel,S,SNRdb)
    else:
        [OFDM_RX,sigma] = ofdm.channel(OFDM_TX,OFDM_TX_piloto,channel,S,SNRdb)

      
    OFDM_RX_ok = ofdm.remove_redund(OFDM_RX, K, N, L, TX_type)
    OFDM_demod = ofdm.DFT(OFDM_RX_ok)
    #OFDM_demod = OFDM_RX_ok

    return OFDM_demod,OFDM_data, OFDM_TX
    
def symb_ber(symbol_block, demapping_table,QAM_est,bits_all):
    N = symbol_block.N
    mu = symbol_block.mu
    bits_all2 = bits_all[0:(N * mu)]
    PS_est = ofdm.Demapping(QAM_est, demapping_table)
    bits_est = ofdm.PS(PS_est)
    ber = np.sum(abs(bits_all2-bits_est))/len(bits_all2)
    return ber

def ensemble(model1,H_LS,scale,N):
    H_hat = model1.predict(np.reshape(H_LS / scale, (1, 2 * N)))
    H_hat_ok = []
    for i in range(N):
        H_hat_ok.append(H_hat[0, i] + 1j * H_hat[0, i + N])
    H_hat_ok = np.asarray(H_hat_ok) * scale
    return H_hat_ok





def symb_transmission2(symbol_block, channelResponse, pilot, nonlinearity):
    S = symbol_block.S
    N = symbol_block.N
    K = symbol_block.K
    L = len(channelResponse)-1
    mu = symbol_block.mu
    mapping_table = symbol_block.mapping_table
    bits_all = symbol_block.bits
    SNRdb = symbol_block.SNRdb

    bits = bits_all[0:(N*mu)]
    bits_SP = ofdm.SP(bits,N,mu)
    QAM = ofdm.Mapping(bits_SP,mapping_table)
    OFDM_data = ofdm.OFDM_symbol_all_sub(N,QAM)
    OFDM_time = ofdm.IDFT(OFDM_data)


    bits_ant = np.random.binomial(n=1, p=0.5, size=(N * mu))
    bits_SP_ant = ofdm.SP(bits_ant, N, mu)
    QAM_ant = ofdm.Mapping(bits_SP_ant, mapping_table)
    OFDM_data_ant = ofdm.OFDM_symbol_all_sub(N, QAM_ant)
    OFDM_time_ant = ofdm.IDFT(OFDM_data_ant)


    OFDM_TX = ofdm.add_redund(OFDM_time, K, N, 'ZP')
    OFDM_TX_ant = ofdm.add_redund(OFDM_time_ant, K, N, 'ZP')

    if nonlinearity==True:
        # OFDM_TX = np.tanh(OFDM_TX)
        # OFDM_TX_ant = np.tanh(OFDM_TX_ant)
        CR = 1.3
        sigma_x = np.sqrt(np.var(OFDM_TX))
        CL = CR * sigma_x
        OFDM_TX = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX]
        CR = 1.3
        sigma_x = np.sqrt(np.var(OFDM_TX_ant))
        CL = CR * sigma_x
        OFDM_TX_ant = [i / abs(i) * CL if abs(i) > CL else i for i in OFDM_TX_ant]


    [OFDM_RX,sigma] = ofdm.channel(OFDM_TX,OFDM_TX_ant,channelResponse,S,SNRdb)

    if pilot == True:
        aux1 = np.eye(N)
        aux2_1 = np.eye(K)
        aux2_2 = np.zeros((N-K,K))
        p2 = np.vstack((aux2_1,aux2_2))
        R_zp = np.hstack((aux1,p2))
        OFDM_RX_ok = np.dot(R_zp,OFDM_RX)

    else:
        OFDM_RX_ok = ofdm.remove_redund(OFDM_RX, K, N, L, 'ZPZJ')



    return OFDM_RX_ok,OFDM_data


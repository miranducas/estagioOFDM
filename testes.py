Python 3.8.5 (default, Jan 27 2021, 15:41:15) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license()" for more information.
>>> bits = np.random.binomial(n=1, p=0.5, size=(10 * 4))
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    bits = np.random.binomial(n=1, p=0.5, size=(10 * 4))
NameError: name 'np' is not defined
>>> import numpy as np
>>> bits = np.random.binomial(n=1, p=0.5, size=(10 * 4))
>>> bits.reshape(10,4)
array([[1, 1, 0, 0],
       [1, 0, 0, 1],
       [1, 1, 0, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 1],
       [1, 0, 0, 1],
       [0, 1, 0, 1],
       [1, 0, 0, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 1]])
>>> def QAM_mod(number_of_symbols):
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

>>> [mapping_table,demapping_table] = ofdm.QAM_mod(2**4)
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    [mapping_table,demapping_table] = ofdm.QAM_mod(2**4)
NameError: name 'ofdm' is not defined
>>> [mapping_table,demapping_table] = QAM_mod(2**4)
>>> bits
array([1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
       0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
>>> bits_SP = bits.reshape(10,4)
>>> bits_SP
array([[1, 1, 0, 0],
       [1, 0, 0, 1],
       [1, 1, 0, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 1],
       [1, 0, 0, 1],
       [0, 1, 0, 1],
       [1, 0, 0, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 1]])
>>> bits_SP.reshape(-1)
array([1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
       0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
>>> bits.reshape(-1)
array([1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
       0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
>>> def Mapping(bits,mapping_table):
    return np.array([mapping_table[tuple(b)] for b in bits])

>>> QAM = Mapping(bits_SP,mapping_table)
>>> QAM
array([ 1.-3.j,  3.-1.j,  1.-1.j, -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,
        3.-3.j, -1.+3.j,  3.-1.j])
>>> bits_SP
array([[1, 1, 0, 0],
       [1, 0, 0, 1],
       [1, 1, 0, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 1],
       [1, 0, 0, 1],
       [0, 1, 0, 1],
       [1, 0, 0, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 1]])
>>> QAM.reshape(-1)
array([ 1.-3.j,  3.-1.j,  1.-1.j, -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,
        3.-3.j, -1.+3.j,  3.-1.j])
>>> QAM
array([ 1.-3.j,  3.-1.j,  1.-1.j, -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,
        3.-3.j, -1.+3.j,  3.-1.j])
>>> np.fft.ifft(QAM)
array([ 0.6       -1.2j       ,  0.9894396 -0.45076606j,
        0.6969175 +0.46043951j, -0.7441311 -0.8023108j ,
       -0.14222601-0.84328816j, -1.2       +0.6j       ,
        0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j])
>>> def OFDM_symbol_all_sub(S,QAM_payload):
    dataCarriers = np.arange(S)
    symbol = np.zeros(S, dtype=complex) # overall S subcarriers
    symbol[dataCarriers] = QAM_payload # allocate the data subcarriers
    return symbol

>>> OFDM_data = ofdm.OFDM_symbol_all_sub(10,QAM)
Traceback (most recent call last):
  File "<pyshell#22>", line 1, in <module>
    OFDM_data = ofdm.OFDM_symbol_all_sub(10,QAM)
NameError: name 'ofdm' is not defined
>>> OFDM_data = OFDM_symbol_all_sub(10,QAM)
>>> OFDM_data
array([ 1.-3.j,  3.-1.j,  1.-1.j, -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,
        3.-3.j, -1.+3.j,  3.-1.j])
>>> QAM
array([ 1.-3.j,  3.-1.j,  1.-1.j, -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,
        3.-3.j, -1.+3.j,  3.-1.j])
>>> OFDM_time = np.fft.ifft(OFDM_data)
>>> OFDM_time
array([ 0.6       -1.2j       ,  0.9894396 -0.45076606j,
        0.6969175 +0.46043951j, -0.7441311 -0.8023108j ,
       -0.14222601-0.84328816j, -1.2       +0.6j       ,
        0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j])
>>> OFDM_time.shape
(10,)
>>> QAM.shape
(10,)
>>> bits_SP.shape
(10, 4)
>>> np.fft.fft(OFDM_time)
array([ 1.-3.j,  3.-1.j,  1.-1.j, -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,
        3.-3.j, -1.+3.j,  3.-1.j])
>>> ef add_redund(OFDM_time, K, N, type):
    if K==0 :
        OFDM_TX = OFDM_time
    else:
        if type == 'CP':
            cp = OFDM_time[-K:]  # take the last CP samples
            OFDM_TX = np.hstack([cp, OFDM_time])

        if type == 'ZP':
            F = np.vstack((np.eye(N), np.zeros((K, N))))
            OFDM_TX = np.dot(F, OFDM_time)
        if type == 'CP-SC':
            aux1 = np.zeros((K,N-K))
            aux2 = np.eye(K)
            aux = np.hstack((aux1,aux2))
            F = np.vstack((aux, np.eye(N)))
            OFDM_TX = np.dot(F, OFDM_time)

    return OFDM_TX
SyntaxError: invalid syntax
>>> def add_redund(OFDM_time, K, N, type):
    if K==0 :
        OFDM_TX = OFDM_time
    else:
        if type == 'CP':
            cp = OFDM_time[-K:]  # take the last CP samples
            OFDM_TX = np.hstack([cp, OFDM_time])

        if type == 'ZP':
            F = np.vstack((np.eye(N), np.zeros((K, N))))
            OFDM_TX = np.dot(F, OFDM_time)
        if type == 'CP-SC':
            aux1 = np.zeros((K,N-K))
            aux2 = np.eye(K)
            aux = np.hstack((aux1,aux2))
            F = np.vstack((aux, np.eye(N)))
            OFDM_TX = np.dot(F, OFDM_time)

    return OFDM_TX

>>> OFDM_TX = add_redund(OFDM_time, 4, 10, 'CP-SC')
>>> OFDM_TX
array([ 0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j,
        0.6       -1.2j       ,  0.9894396 -0.45076606j,
        0.6969175 +0.46043951j, -0.7441311 -0.8023108j ,
       -0.14222601-0.84328816j, -1.2       +0.6j       ,
        0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j])
>>> OFDM_TX.shape
(14,)
>>> OFDM_TX = add_redund(OFDM_time, 4, 10, 'CP')
>>> OFDM_TX
array([ 0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j,
        0.6       -1.2j       ,  0.9894396 -0.45076606j,
        0.6969175 +0.46043951j, -0.7441311 -0.8023108j ,
       -0.14222601-0.84328816j, -1.2       +0.6j       ,
        0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j])
>>> OFDM_TX2 = add_redund(OFDM_time, 4, 10, 'CP-SC')
>>> OFDM_TX == OFDM_TX2
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True])
>>> def remove_redund(OFDM_RX, K, N, L, type):

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

        if type == 'CP-SC':
            G = np.hstack((np.zeros((N,K)),np.eye(N)))
            result = np.dot(G,OFDM_RX) 
            
    return result

>>> OFDM_RX_ok = ofdm.remove_redund(OFDM_TX, K, N, L, 'CP')
Traceback (most recent call last):
  File "<pyshell#44>", line 1, in <module>
    OFDM_RX_ok = ofdm.remove_redund(OFDM_TX, K, N, L, 'CP')
NameError: name 'ofdm' is not defined
>>> OFDM_RX_ok = remove_redund(OFDM_TX, K, N, L, 'CP')
Traceback (most recent call last):
  File "<pyshell#45>", line 1, in <module>
    OFDM_RX_ok = remove_redund(OFDM_TX, K, N, L, 'CP')
NameError: name 'K' is not defined
>>> OFDM_RX_ok = remove_redund(OFDM_TX, 4, 10, 4, 'CP')
>>> OFDM_RX_ok2 = remove_redund(OFDM_TX2, 4, 10, 4, 'CP-SC')

>>> OFDM_RX_ok
array([ 0.6       -1.2j       ,  0.9894396 -0.45076606j,
        0.6969175 +0.46043951j, -0.7441311 -0.8023108j ,
       -0.14222601-0.84328816j, -1.2       +0.6j       ,
        0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j])
>>> OFDM_RX_ok2
array([ 0.6       -1.2j       ,  0.9894396 -0.45076606j,
        0.6969175 +0.46043951j, -0.7441311 -0.8023108j ,
       -0.14222601-0.84328816j, -1.2       +0.6j       ,
        0.61861921+0.44328816j, -0.27390289-0.89211639j,
        0.2266893 -0.86043951j,  0.22859439+0.54519325j])
>>> OFDM_RX_ok == OFDM_RX_ok2
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])
>>>  OFDM_TX = ofdm.add_redund(OFDM_data, 4, 10, 'CP-SC')
 
SyntaxError: unexpected indent
>>> OFDM_TX = ofdm.add_redund(OFDM_data, 4, 10, 'CP-SC')
Traceback (most recent call last):
  File "<pyshell#52>", line 1, in <module>
    OFDM_TX = ofdm.add_redund(OFDM_data, 4, 10, 'CP-SC')
NameError: name 'ofdm' is not defined
>>> OFDM_TX = add_redund(OFDM_data, 4, 10, 'CP-SC')
>>> OFDM_TX
array([-1.-1.j,  3.-3.j, -1.+3.j,  3.-1.j,  1.-3.j,  3.-1.j,  1.-1.j,
       -3.-3.j, -3.-1.j,  3.-1.j, -1.-1.j,  3.-3.j, -1.+3.j,  3.-1.j])
>>> bits_pilot = np.random.binomial(n=1, p=0.5, size=(10 * 4));
>>> bits_pilot_SP = bits_pilot.reshape(10,4)
>>> bits_pilot_SP
array([[1, 0, 1, 1],
       [1, 0, 1, 0],
       [1, 0, 1, 0],
       [0, 1, 1, 1],
       [1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 0, 1],
       [1, 0, 0, 1]])
>>> QAM_p = Mapping(bits_pilot_SP, mapping_table)
>>> QAM_p
array([ 3.+1.j,  3.+3.j,  3.+3.j, -1.+1.j,  3.-3.j,  1.-3.j,  1.-3.j,
        1.+3.j,  1.-1.j,  3.-1.j])
>>> OFDM_data_p = OFDM_symbol_all_sub(10,QAM_p)
>>> OFDM_TX_p = ofdm.add_redund(OFDM_data_p, 4, 10, 'CP-SC')
Traceback (most recent call last):
  File "<pyshell#61>", line 1, in <module>
    OFDM_TX_p = ofdm.add_redund(OFDM_data_p, 4, 10, 'CP-SC')
NameError: name 'ofdm' is not defined
>>> OFDM_TX_p = add_redund(OFDM_data_p, 4, 10, 'CP-SC')
>>> OFDM_TX_p
array([ 1.-3.j,  1.+3.j,  1.-1.j,  3.-1.j,  3.+1.j,  3.+3.j,  3.+3.j,
       -1.+1.j,  3.-3.j,  1.-3.j,  1.-3.j,  1.+3.j,  1.-1.j,  3.-1.j])
>>> OFDM_TX_p == OFDM_TX
array([False, False, False,  True, False, False, False, False, False,
       False, False, False, False,  True])
>>> def channel(signal,signal_ant,channelResponse,S,SNRdb):
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


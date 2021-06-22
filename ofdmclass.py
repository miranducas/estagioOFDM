import numpy as np
import ofdm
################################################################################################################################
# This script 
################################################################################################################################
class symbol_block:
    def __init__(self, bits, N, K, mu, mapping_table, SNRdb):
        self.S = N + K
        self.N = N
        self.K = K
        self.bits = bits
        self.mu = mu
        self.mapping_table = mapping_table
        self.SNRdb = SNRdb

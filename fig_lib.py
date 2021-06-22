import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pathlib as pb
import os
#import plotly.graph_objects as go
import numpy as np


################################################################################################################################
# This script is used to generate the BER curve by using the simulation results in /results_ber folder.
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------#
def proc_nan(ber,ind):
    ber = np.asarray(ber)
    ber = ber[ind]
    ber[ber == 0] = np.nan
    return ber


def BER_SNR(folder_result):

# Going to results folder and selecting ber results files:
    

    path_ber = pb.Path.cwd()/folder_result
    os.chdir(folder_result)

    list_ber = [file for file in pb.os.listdir(path_ber) if file.endswith('.pickle')]
    output_filename = list_ber[0].split('_snr')[0]

    SNRdb_range = []

    ber_wcp=[]
    ber_wcp_exact=[]
    ber_ls=[]
    ber_zf_mr=[]
    ber_net1_wcp=[]
    ber_net1_ncp=[]
    ber_net2_ncp=[]
    ber_net1_cp10 = []
    ber_net1_wzp = []
    ber_net2_wzp =[]
    ber_wzpzj = []
    ber_net1_zp10 = []

#------------------------------------------------------------------------------------------------------------------------------#
    # Reading ber files:

    for dt in list_ber: 
        snr = dt.split('_snr_')[1]
        SNRdb_range.append(int(snr.split('.')[0]))
        with open(dt, 'rb') as f:
            #ber_ls1, ber_zf_mr1, ber_wcp1, ber_wcp_exact1  = pickle.load(f)
            ber_ls1, ber_zf_mr1, ber_wcp1, ber_wcp_exact1, ber_net1_wcp1, ber_net1_ncp1, ber_net2_ncp1, ber_net1_cp101  = pickle.load(f)
            #ber_ls1, ber_zf_mr1, ber_wcp1, ber_wcp_exact1, ber_net1_wcp1, ber_net1_ncp1, ber_net2_ncp1, ber_net1_cp101, ber_net1_wzp1, ber_wzpzj1, ber_net1_zp101 = pickle.load(f)
    
        
        ber_ls.append(ber_ls1)
        ber_wcp.append(ber_wcp1)
        ber_wcp_exact.append(ber_wcp_exact1)
        ber_zf_mr.append(ber_zf_mr1)
        ber_net1_wcp.append(ber_net1_wcp1)
        ber_net1_ncp.append(ber_net1_ncp1)
        ber_net2_ncp.append(ber_net2_ncp1)
        ber_net1_cp10.append(ber_net1_cp101)


        #ber_net1_wzp.append(ber_net1_wzp1)
        #ber_wzpzj.append(ber_wzpzj1)
        #ber_net1_zp10.append(ber_net1_zp101)
 #------------------------------------------------------------------------------------------------------------------------------#
    # Converting to array:

    SNRdb_range = np.asarray(SNRdb_range)
    ind = np.argsort(SNRdb_range)
    SNRdb_range = SNRdb_range[ind]


    ber_ls = proc_nan(ber_ls,ind)
    ber_wcp = proc_nan(ber_wcp,ind)
    ber_wcp_exact = proc_nan(ber_wcp_exact,ind)
    ber_zf_mr = proc_nan(ber_zf_mr,ind)
    ber_net1_wcp = proc_nan(ber_net1_wcp,ind)
    ber_net1_ncp = proc_nan(ber_net1_ncp,ind)
    ber_net2_ncp = proc_nan(ber_net2_ncp,ind)
    ber_net1_cp10 = proc_nan(ber_net1_cp10,ind)
    
    
    #ber_net1_wzp= proc_nan(ber_net1_wzp,ind)
    #ber_wzpzj = proc_nan(ber_wzpzj,ind)
    #ber_net1_zp10= proc_nan(ber_net1_zp10,ind)
    print(str(SNRdb_range))
   
    os.chdir("..")
    #return SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact
    return SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10
    #return SNRdb_range, ber_ls, ber_zf_mr, ber_wcp, ber_wcp_exact,  ber_net1_wcp, ber_net1_ncp, ber_net2_ncp, ber_net1_cp10, ber_net1_wzp, ber_wzpzj, ber_net1_zp10

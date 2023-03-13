# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 13:15:27 2021

@author: nbrow
"""
import numpy as np
import matplotlib.pyplot as plt
import random
Iterations=500
fig,ax=plt.subplots()
for It in range(1,Iterations+1):

    #FileName_C='UC_Design_AR3_T_Trial{}'.format(Num)
    FileName_T='UC_Design_NL_{}'.format(It)
    Results_C=np.load('Result_Files/'+FileName_T+'.npy')
    if Results_C[-1,0]<1e-4 and np.max(Results_C[:,1])*10**3<2 and random.randint(0,10)==1:
        #plt.legend(loc='best',fontsize=20)
        
        ax.plot(Results_C[:,0]*10**3,Results_C[:,1]*10**3,'-',label='Trial_{}'.format(It))
        ax.set_xlabel('Displacement (E-3) [mm]')
        ax.set_ylabel('Resultant Force [kN]')
        ax.set_ylim([0,2])
        ax.set_xlim([0,10])
        ax.legend()
        #print(Results_C[-1,1])
        fig2,ax2= plt.subplots()
        Plot=np.load('ML_Input_Files/UC_Design_{}.npy'.format(It))
        ax2.imshow(Plot,cmap='Blues',origin='lower')
        ax2.axis('off')
    #plt.legend()
        




    
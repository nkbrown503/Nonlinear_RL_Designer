# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:33:14 2022

@author: nbrow
"""


import numpy as np
from gym import Env
import copy 
from gym.spaces import Discrete, Box
import random
import sys
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\RL_NL_Design\Unit_Cell_Designing\VAE')
from autoencoder import VAE, Autoencoder
from RL_Bezier import RL_Bezier_Design as RLBD
from Matrix_Transforms import isolate_largest_group_original
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\RL_NL_Design\Unit_Cell_Designing\Surrogate_Model')

from Resnet_model import create_res_net
import math
import matplotlib.pyplot as plt 
import argparse
np.random.seed(1)
def parse_args(End_Nodes):
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--loops",type=int,default=4)
    parser.add_argument("--filters",type=int,default=32)
    parser.add_argument("--Trial_Num",type=int,default=1)
    parser.add_argument("--end_nodes",type=int,default=End_Nodes)
    
    args=parser.parse_args()
    return args


class UC_Env(Env):
    def __init__(self):
        #Action Space [Starting Corner, Ending Corner, Y Coor of IP1, X Coor of IP2, Y Coor of IP2, X Coor of IP2, Thickness]
        self.action_space=Box(np.array([0,0,0,0,0,0,0]),np.array([1,1,1,1,1,1,1]))
        self.max_steps=6
        self.E_Y=20
        self.E_X=60
        self.args_L=parse_args(11) 
        self.args_E=parse_args(1)
        self.Legal=False

        #The following values are used to standardize the FD curves and/or the Latent Space 

        #Import the Autoencoder needed to define the latent space of each unit cell
        _,self.VAE_Encoder,_,_,_=VAE(self.E_Y,self.E_X,num_channels=1,latent_space_dim=24)
        self.VAE_Encoder.load_weights("../VAE/VAE_encoder_24.h5") 
        
        #Import the surrogate model to predict the FD curve of each unit cell 
        self.surrogate_model_loading= create_res_net(self.args_L) 
        self.surrogate_model_energy= create_res_net(self.args_E)
        self.surrogate_model_loading.load_weights('../Surrogate_Model/checkpoints/UC_NL_Loading_RNN_Model_Weights/cp.ckpt')
        self.surrogate_model_energy.load_weights('../Surrogate_Model/checkpoints/UC_NL_Energy_RNN_Model_Weights/cp.ckpt')

    def step(self,action):

        self.state_UC_=copy.deepcopy(self.state_UC)
        
        #Add material in the shape of a Bezier curve design according to the action from the RL agent 
        self.state_UC=RLBD(action,self.state_UC)
        try:
            self.Current_Force_=copy.deepcopy(self.Current_Force)
        except:
                'Nothing'
        self.obs_=copy.deepcopy(self.obs)
        
        #Take the unit cell and produce the 48 dimensional latent space 
 
        self.obs_[11:]=np.reshape(self.VAE_Encoder.predict(np.reshape(self.state_UC,(1,20,60,1)),verbose=0),(24,))

        self.step_count+=1
        SingleCheck=isolate_largest_group_original(self.state_UC)

        #Check if the unit cell is legal
        if SingleCheck[1]==False and self.step_count>1:
            Reward=-1
            Done=False
            self.Legal=False
            self.Perc_Error=1
            self.Force_Error=1
            self.Perc_Error2=1
            self.Energy=1
        elif SingleCheck[1]==False and self.step_count==1:
            Reward=0
            self.Force_Error=1
            Done=False
            self.Legal=False
            self.Perc_Error=1
            self.Perc_Error2=1
            self.Energy=1
            
        else:
            self.Legal=True
            self.state_UC=np.reshape(self.state_UC,(20,60))
            self.surrogate_UC=np.zeros((2*self.E_Y,2*self.E_X))
            self.surrogate_UC[:self.E_Y,:self.E_X]=self.state_UC
            self.surrogate_UC[:self.E_Y,self.E_X:]=np.flip(self.state_UC,axis=1)
            self.surrogate_UC[self.E_Y:,:self.E_X]=np.flip(self.state_UC,axis=0)
            self.surrogate_UC[self.E_Y:,self.E_X:]=np.flip(np.flip(self.state_UC,axis=0),axis=1)
            self.surrogate_UC=np.reshape(self.surrogate_UC,(1,40,120,1))
            
            #Predict the FD curve accordinging to the latent space of the unit cell
            self.Loading_Force=self.surrogate_model_loading.predict(self.surrogate_UC,verbose=0)[0]
            self.Energy=self.surrogate_model_energy.predict(self.surrogate_UC,verbose=0)
         
            #self.Current_Force[1:]=(self.Current_Force[1:]*self.Force_stdev[1:])+self.Force_Mean[1:]

            #Compare the error between the Desired FD and the True FD 
            self.Force_Error=(np.max([abs((i-j)) for i,j in zip(self.Loading_Force[1:],self.Desired_Force[1:])]))
            self.Perc_Error=(np.mean([abs((i-j)/j) for i,j in zip(self.Loading_Force[4:],self.Desired_Force[4:])]))
            self.Perc_Error2=(np.mean([abs((i-j)/i) for i,j in zip(self.Loading_Force[4:],self.Desired_Force[4:])]))
    

            Reward=np.max([-self.Perc_Error,-self.Perc_Error2,-1])
            print(Reward)
            Reward+=(1-self.Energy)
            #Reward=np.max([-self.Force_Error,-1])
            if self.Perc_Error<0.075 or self.Perc_Error2<0.075:
                #if self.Force_Error<0.025:    #If the percent error is less than 10% than the design is considered satisfactory 
                Done=True
                Reward=1+np.max([-self.Perc_Error,-self.Perc_Error2])
                
            else: 
                Done=False

        if self.step_count>=self.max_steps:
            Done=True 

        self.obs=copy.deepcopy(self.obs_)
        return self.obs_, Reward, Done, self.Legal
            
    def render(self,Legal,i,ax2):
        
        #Reformat 20x60 design domain into the 40x120 unit cell for plotting 
        self.state_UC=np.reshape(self.state_UC,(20,60))
        self.Element_Plot=np.zeros((self.E_Y*2,self.E_X*2))
        self.Element_Plot[0:self.E_Y,0:self.E_X]=self.state_UC
        self.Element_Plot[0:self.E_Y,self.E_X:2*self.E_X]=np.flip(self.state_UC,axis=1)
        self.Element_Plot[self.E_Y:self.E_Y*2,0:self.E_X]=np.flip(self.state_UC,axis=0)
        self.Element_Plot[self.E_Y:self.E_Y*2,self.E_X:2*self.E_X]=np.flip(np.flip(self.state_UC,axis=0),axis=1)
        self.Strain_val=np.linspace(0,1,11)
        
    
        #print the proposed unit cell 
        fig,ax= plt.subplots()
        ax.imshow(self.Element_Plot,cmap='Blues',origin='lower')
        ax.axis('off')

        fig2,ax2=plt.subplots()
        C=['#8B7765','#CD3333','#473C8B','#6CA6CD','#CD4F39','#458B74','#CD69C9','#8E8E38']
        ax2.plot(self.Strain_val,self.Desired_Force,'--',color='#00008B'.format(C[i]),label='Desired Response')
        ax2.set_xlabel('Normalized Displacement',fontsize=20)
        ax2.set_ylabel('Normalized Force',fontsize=20)
        ax2.tick_params(axis='both',labelsize=20)
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])
        #ax2.legend(fontsize=16)
        if self.Legal:
            
            #Print a comparison between the desired and current FD curves 
            ax2.plot(self.Strain_val,self.Loading_Force,'-',color='#00008B'.format(C[i]),label='Current Response')
            #ax2.legend(fontsize=16)

            ax2.set_xlabel('Normalized Displacement',fontsize=20)
            ax2.set_ylabel('Normalized Force',fontsize=20)
            ax2.tick_params(axis='both',labelsize=20)
            
        
    def reset(self,Test,i):
        
        self.step_count=0
        #Reset the unit cell and the RL observation
        self.state_UC=np.zeros((20,60))
        self.state_UC[0,0]=1
        self.state_UC[-1,0]=1
        self.state_UC[0,-1]=1
        self.state_UC[-1,-1]=1
        self.obs=np.zeros((35,)) #Top 11 Values are desired force response and bottom 48 are current latent space


        self.Desired_Force=[0.25*x**2.4 for x in np.arange(0,1.1,0.1)]

        
        self.obs[:11]=np.reshape(self.Desired_Force,(11,))
    
        return self.obs
    
        

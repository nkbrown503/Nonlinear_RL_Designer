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
sys.path.insert(0,r'/scratch/nkbrown/nkbrown/VAE')

from autoencoder import VAE, Autoencoder
from RL_Bezier import RL_Bezier_Design as RLBD
from Matrix_Transforms import are_ones_continuous
from scipy.optimize import curve_fit
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\RL_NL_Design\Unit_Cell_Designing\Surrogate_Model')
sys.path.insert(0,r'/scratch/nkbrown/nkbrown/NL_Surrogate_Model')

from Surrogate_Model import CNN_model
from utils import PCA_inverse, PCA_transform,curve_func

import math
import matplotlib.pyplot as plt 
import argparse
np.random.seed(12)
def parse_args(): #Additional hyperparameters of the DRL agent 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--loops",type=int,default=4)
    parser.add_argument("--filters",type=int,default=32)
    parser.add_argument("--Trial_Num",type=int,default=1)
    parser.add_argument("--Filter_Start",type=int,default=124)
    parser.add_argument("--FCC_Nodes",type=int,default=152)
    parser.add_argument("--Filter_Increase",type=float,default=1.463)
    parser.add_argument("--end_nodes",type=int,default=3)  
  
    args=parser.parse_args()
    return args


class UC_Env(Env):
    def __init__(self):
        #Action Space [Starting Corner, Ending Corner, Y Coor of IP1, X Coor of IP2, Y Coor of IP2, X Coor of IP2, Thickness]
        self.action_space=Box(np.array([0,0,0,0,0,0,0]),np.array([1,1,1,1,1,1,1]))
        self.max_steps=6 #The agent is only allow 6 steps because too many steps resuts in high volume fraction unit cells 
        self.E_Y=20
        self.E_X=60
        self.PC_Matching=np.load('Constants/PCA_Outputs.npy')
        self.Legal=False
        self.args=parse_args()
        self.Training_Set=np.load('Constants/PC_Training_Set.npy')
        self.Training_Curves=np.load('Constants/Training_SS_Curves.npy')
        self.E_TPU=np.load('Constants/E_TPU_Fitted_Data.npy') #Experimental results of the E-TPU in cyclic compression
        self.Avg_Reward_Coef=np.load('Constants/Reward_Func_Coef.npy')[:,0] #The following results are all used to define the energy return reward function
        self.Upper_Bonus_Coef=np.load('Constants/Reward_Func_Coef.npy')[:,1]
        self.Lower_Bonus_Coef=np.load('Constants/Reward_Func_Coef.npy')[:,2]
        self.Upper_Reward_Coef=np.load('Constants/Reward_Func_Coef.npy')[:,3]
        self.Lower_Reward_Coef=np.load('Constants/Reward_Func_Coef.npy')[:,4]
        #The following values are used to standardize the Stress-strain curves or PCA directions
        self.selected_eigenvectors=np.load('Constants/PCA_Eigenvectors.npy')
        self.mean_data=np.load('Constants/PCA_mean_data.npy') #The following values are the standardizing values for the PCA 
        self.PCA_min=np.load('Constants/PCA_min.npy')
        self.PCA_max=np.load('Constants/PCA_max.npy')
        self.Test_Set_1=[1900, 2036, 4745, 4879, 5720, 7192, 5394,    2,  369,  988, 1545,
               4443, 4994, 7008, 6106,  725, 4547, 7335, 4532, 1980, 2167, 5155,
               6613, 6251, 3755,    0,   46, 6437, 6102, 6216, 7986, 4283, 4114,
                105,  114, 7058, 7454, 7083, 4755, 5597, 6299, 7204, 7469, 5919,
                 46,  124, 6437, 4596, 6778, 5627, 6060, 6791, 4299,  755, 6621,
               3758, 6029, 5603, 6181, 7000,  944, 1274, 1865, 5572,  625, 1937,
               6170, 4924, 5042, 7119, 1101, 2156, 3253, 4799,   46, 1028, 1880,
               4456,  885, 6505, 2336, 1188, 2856, 4191, 6117, 4615,   39] #This test set was specifically used for the publication as it gave a diverse range of high-performing repsonses 
        
 
        #The Principal directions of the E-TPU
        self.E_TPU_PD=PCA_transform(self.E_TPU,self.selected_eigenvectors,self.mean_data)
        #Import the Autoencoder needed to define the latent space of each unit cell
        _,self.VAE_Encoder,_,_,_=VAE(self.E_Y,self.E_X,num_channels=1,latent_space_dim=24)
        self.VAE_Encoder.load_weights("../VAE/VAE_encoder_24.h5") 
        
        #Import the surrogate model to predict the principal directions for each unit cell which corresponds
        #to a respective stress-strain curve 
        self.surrogate_model=CNN_model(self.args)
        self.surrogate_model.load_weights('UC_NL_CNN_Model_Weights/cp.ckpt')

    def step(self,action):

        self.state_UC_=copy.deepcopy(self.state_UC)
        
        #Add material in the shape of a Bezier curve design according to the action from the RL agent 
        self.state_UC=RLBD(action,self.state_UC)

        self.obs_=copy.deepcopy(self.obs)
        
        #Take the unit cell and produce the 48 dimensional latent space 
 
        self.obs_[4:]=np.reshape(self.VAE_Encoder.predict(np.reshape(self.state_UC,(1,20,60,1)),verbose=0),(24,))

        self.step_count+=1
        SingleCheck=are_ones_continuous(self.state_UC)

        #Check if the unit cell is legal
        if SingleCheck==False and self.step_count>1:
            Reward=-2
            Done=False
            self.Legal=False
            self.Loading_Error=1

        elif SingleCheck==False and self.step_count==1:
            Reward=0
            Done=False
            self.Legal=False
            self.Loading_Error=1
            
        else:
            self.Legal=True
            #Double mirror the design domain to produce the unit cell
            self.state_UC=np.reshape(self.state_UC,(20,60))
            self.surrogate_UC=np.zeros((2*self.E_Y,2*self.E_X))
            self.surrogate_UC[:self.E_Y,:self.E_X]=self.state_UC
            self.surrogate_UC[:self.E_Y,self.E_X:]=np.flip(self.state_UC,axis=1)
            self.surrogate_UC[self.E_Y:,:self.E_X]=np.flip(self.state_UC,axis=0)
            self.surrogate_UC[self.E_Y:,self.E_X:]=np.flip(np.flip(self.state_UC,axis=0),axis=1)
            
            #Tesselate the unti cell into the 3x3 metamaterial to serve as the input to the surrogate model
            self.surrogate_Tess=np.zeros((120,540))
            for y_move in range(0,3):
                for x_move in range(0,4):
                    if y_move%2==0:
                        self.surrogate_Tess[(40)*y_move:(40)*(y_move+1),(120)*(x_move):(120)*(x_move+1)]=self.surrogate_UC
                    else:
                        self.surrogate_Tess[(40)*y_move:(40)*(y_move+1),(120)*(x_move)+60:(120)*(x_move+1)+60]=self.surrogate_UC

            self.surrogate_Tess=self.surrogate_Tess[0:120,60:-120]
            self.surrogate_Tess=np.reshape(self.surrogate_Tess,(1,120,360,1))
            
            #Predict the principal directions accordinging to the 3x3 tessellation of the unit cell
            self.Current_PD=self.surrogate_model.predict(self.surrogate_Tess,verbose=0)[0]
            
            self.Current_PD=(((self.Current_PD+1)/2)*(self.PCA_max-self.PCA_min))+self.PCA_min
            self.Current_SS=PCA_inverse(self.Current_PD,self.selected_eigenvectors,self.mean_data)

            #Extract Loading and Unloading stresses and strains from inversed srrogate predictions 
            L_Stress=self.Current_SS[:21]
            U_Stress=self.Current_SS[21:42]
            L_Strain=self.Current_SS[42:63]
            U_Strain=self.Current_SS[63:]
            #Compare the error between the Desired FD and the True FD 

            #self.PD_Error=(np.mean([abs((i-j)/j) for i,j in zip(self.Current_PD,self.Desired_PD)]))
            self.Loading_Error=(np.mean([abs((i-j)/j) for i,j in zip(self.Current_SS[7:21],self.New_L_Stress[7:])]))
            self.Loading_Error2=(np.mean([abs((i-j)/i) for i,j in zip(self.Current_SS[7:21],self.New_L_Stress[7:])]))
            #self.Perc_Error2=(np.mean([abs((i-j)/i) for i,j in zip(self.Loading_Force[3:],self.Desired_Force[3:])]))
            
            #Calculate the energy return 
            Loading_E=sum([((L_Stress[x+1]+L_Stress[x])/2)*(L_Strain[x+1]-L_Strain[x]) for x in range(0,len(L_Stress)-1)])
            Unloading_E=sum([abs(((U_Stress[x+1]+U_Stress[x])/2)*(U_Strain[x+1]-U_Strain[x])) for x in range(0,len(U_Stress)-1)])

            self.Energy_Return=(Unloading_E/Loading_E)
            if self.Energy_Return>1:
                self.Energy_Return=1
            #Reward for the difference between desired and resulting deformation responses 
            Reward=np.max([-self.Loading_Error,-self.Loading_Error2,-2])
            
            
            if self.obs[0]==0: #Reward for energy return given hysteresis minimization 
                Reward+=-np.max([(self.Energy_Return-self.Lower_Reward)*self.Energy_Scalar,0])
                if self.Loading_Error<0.1 and self.Energy_Return<=self.Lower_Bonus:
                    Done=True
                    Reward=1-self.Loading_Error
                else: 
                    Done=False
            else:
                #Reward for energy return given hysteresis maximization
                Reward+=-np.max([(self.Upper_Reward-self.Energy_Return)*self.Energy_Scalar,0])
                if self.Loading_Error<0.1 and self.Energy_Return>=self.Upper_Bonus:
                    Done=True
                    Reward=1-self.Loading_Error
                else: 
                    Done=False

        if self.step_count>=self.max_steps:
            Done=True 
        self.obs=copy.deepcopy(self.obs_)

        return self.obs_, Reward, Done, self.Legal
            
    def render(self,Legal,i,ax2,Turn):
        #Rendering shows the resulting unit cell and the comparison of the desired and resulting responses 
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
        fig,ax= plt.subplots()
        #ax.imshow(self.Element_Plot[:20,:60],cmap='Blues',origin='lower')
        #ax.axis('off')
        plt.figure()
        C=['#6495ED','#473C8B']
        Label=['RL Max Hyst.','RL Min Hyst.']

        if self.Legal:
            
            plt.plot(self.Current_SS[42:63],(self.Current_SS[:21]),'--',color=C[Turn],linewidth=4,label=Label[Turn])
            plt.plot(self.Current_SS[63:],(self.Current_SS[21:42]),'--',color=C[Turn],linewidth=4)
        
        plt.plot(self.Desired_SS[42:63],self.Desired_SS[:21],':',dashes=(1,0.5),color='#CD3333',label='Desired',linewidth=4)
        plt.plot(self.Desired_SS[63:],self.Desired_SS[21:42],':',dashes=(1,0.5),color='#CD3333',linewidth=4)
        plt.ylim([0,self.Desired_SS[21]*1.25])
        plt.legend(fontsize=16)
        plt.xlim([0,0.21])
        plt.ylim([0,0.08])
        plt.xlabel('Strain [-]',fontsize=16)
        plt.ylabel('Stress [MPa]',fontsize=16)
        plt.xticks([0,0.05,0.1,0.15,0.2],fontsize=16)
        plt.yticks([0,0.02,0.04,0.06,0.08],fontsize=16)
        
        

    def reset(self,Test,i):
        
        self.step_count=0
        #Reset the unit cell and the RL observation
        self.state_UC=np.zeros((20,60))
        self.state_UC[0,0]=1
        self.state_UC[-1,0]=1
        self.state_UC[0,-1]=1
        self.state_UC[-1,-1]=1
        self.obs=np.zeros((28,)) #Top 2 Values are principal directions corresponding to desired FD response and bottom 24 are current latent space
        if not Test: #The first value of the observation is a boolena variable to min or  max hysteresis 
            self.obs[0]=np.random.choice([0,1])
        if not Test:
            self.Desired_SS=self.Training_Curves[np.random.choice(self.Training_Set),:] #Randomly select one of the training curve desired responses
            self.Desired_PD=PCA_transform(self.Desired_SS,self.selected_eigenvectors,self.mean_data)
        else:
            #Currently testing with the desired response as the E-TPU, alternatively you can use a testing set of desired curves 
            self.Desired_PD=self.E_TPU_PD
            self.Desired_SS=PCA_inverse(self.Desired_PD,self.selected_eigenvectors,self.mean_data)
            self.Desired_SS=self.E_TPU 
            #self.Desired_SS=self.Test_Set_1[np.random.randint(0,len(self.Test_Set_1))]
    
        #The 2nd-4th values of the observation are the PCA of the desired response 
        self.obs[1:4]=self.Desired_PD

        #Extract loading and unloading stress/strain values from the results files         
        self.New_L_Stress=self.Desired_SS[:21]
        #Set the upper and lower bounds of when the agent should receive additional reward 
        self.Upper_Bonus=curve_func(self.New_L_Stress[-1],*self.Upper_Bonus_Coef)
        self.Lower_Bonus=curve_func(self.New_L_Stress[-1],*self.Lower_Bonus_Coef)
        self.Avg_Return=curve_func(self.New_L_Stress[-1],*self.Avg_Reward_Coef)
        self.Upper_Reward=curve_func(self.New_L_Stress[-1],*self.Upper_Reward_Coef)
        self.Lower_Reward=curve_func(self.New_L_Stress[-1],*self.Lower_Reward_Coef)
        #Define the scaling factor for the energy return reward 
        self.Energy_Scalar=0.0364/((self.Upper_Reward-self.Avg_Return)/3)

        return self.obs

        

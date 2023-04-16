
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:36:56 2023

@author: nbrow
"""
import sys
import os
import argparse
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import random
import matplotlib.pyplot as plt 
from Resnet_model import create_res_net
from Surrogate_Model import FCC_model
sys.path.insert(0,r'/scratch1/nkbrown/VAE')
from autoencoder import VAE, Autoencoder
from tensorflow import random as tf_rand
tf_rand.set_seed(1234)
def func(x,a,b,c,d,e):
    return (a*x**4)+(b*x**3)+(c*x**2)+(d*x)+e


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--layers",type=int,default=3)
    parser.add_argument("--L1_nodes",type=int,default=256)
    parser.add_argument("--LF_nodes",type=int,default=64)
    parser.add_argument("--nodes",type=int,default=128)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--Trial_Num",type=int,default=1)
    
    args=parser.parse_args()
    return args

args=parse_args()   
model = FCC_model(args) # or create_plain_net()
_,VAE_Encoder,VAE_Decoder,_,_=VAE(20,60,1,24)
VAE_Encoder.load_weights("../VAE/VAE_encoder_24.h5")
#Select what you would like to do  Train, Test, or Config 
Action='Test_RNN'


save_name = 'UC_NL_Load_RNN_Model_Weights' # or 'cifar-10_plain_net_30-'+timestr
load_name = "UC_NL_Load_RNN_Model_Weights"
load_path="checkpoints/"+load_name+"/cp.ckpt"
checkpoint_path = "checkpoints/"+save_name+'/cp.ckpt'



X_All=np.load('ML_Noise_Input_Files/ML_Noise_Input_Files.npy') #All the latent spaces 
Y_All_Load=np.load('ML_Noise_Output_Files/ML_Noise_Output_Loading_Files.npy') #4th order polynomial parameters defining loading curve 
Y_All_Unload=np.load('ML_Noise_Output_Files/ML_Noise_Output_Unloading_Files.npy')#4th order polynomial parameters defining unloading curve 
Y_All_Energy=np.load('ML_Noise_Output_Files/ML_Noise_Output_Energy_Files.npy')
X_RN=np.load('ML_Input_Files/Original_Inputs.npy')
Y_RN_Loading=np.load('ML_Output_Files/Original_Loading_Curve_Outputs.npy')
Y_RN_Unloading=np.load('ML_Output_Files/Original_Unloading_Curve_Outputs.npy')

X_Val=np.load('ML_Noise_Input_Files/ML_Noise_Val_Input_Files.npy') #All the latent spaces 
Y_Val_Load=np.load('ML_Noise_Output_Files/ML_Noise_Val_Output_Loading_Files.npy') #4th order polynomial parameters defining loading curve 
Y_Val_Unload=np.load('ML_Noise_Output_Files/ML_Noise_Val_Output_Unloading_Files.npy')#4th order polynomial parameters defining unloading curve 
Y_Val_Energy=np.load('ML_Noise_Output_Files/ML_Noise_Val_Output_Energy_Files.npy')


#Reshape to Configure to Surrogate Model
#X_All=np.reshape(X_All,(np.shape(X_All)[0],np.shape(X_All)[1]))
#Y_All_Load=np.reshape(Y_All_Load,(np.shape(Y_All_Load)[0],np.shape(Y_All_Load)[1]))
#Y_All_Unload=np.reshape(Y_All_Unload,(np.shape(Y_All_Unload)[0],np.shape(Y_All_Unload)[1]))

#X_Val=np.reshape(X_Val,(np.shape(X_Val)[0],np.shape(X_Val)[1]))
#Y_Val_Load=np.reshape(Y_Val_Load,(np.shape(Y_Val_Load)[0],np.shape(Y_Val_Load)[1]))
#Y_Val_Unload=np.reshape(Y_Val_Unload,(np.shape(Y_Val_Unload)[0],np.shape(Y_Val_Unload)[1]))
if Action=='Train': 
    wandb.init(project='NonlinearFD_Surrogate_Training',
           config=vars(args),
           name='Trial_{}'.format(args.Trial_Num))
    history=model.fit(
        x=X_All,
        y=Y_All_Load,
        epochs=150,
        verbose='auto',
        validation_data=(X_Val,Y_Val_Load),
        batch_size=128,
        callbacks=[WandbMetricsLogger()])
    Saved=False
    while Saved==False:
        try:
            model.save_weights(checkpoint_path)
            Saved=True
        except:
            'Nothing'
    plt.plot(history.history['loss'],label='Training Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(save_name+'_LossPlot.png')
    #wandb.log({'Training Loss': min(history.history['loss']),'Validation Loss': min(history.history['val_loss'])})

    
elif Action=='Train_RNN':
    from Resnet_model import create_res_net
    RNmodel=create_res_net()
    wandb.init(project='NonlinearFD_Surrogate_Training',
       config=vars(args),
       name='RNN_{}'.format(args.Trial_Num))
    history=RNmodel.fit(x=X_RN,
                      y=Y_RN_Loading,
                      epochs=150,
                      verbose='auto',
                      validation_split=0.2,
                      batch_size=128,
                      callbacks=[WandbMetricsLogger()])
    Saved=False
    while Saved==False:
        try:
            RNmodel.save_weights(checkpoint_path)
            Saved=True
        except:
            'Nothing'
    plt.plot(history.history['loss'],label='Training Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(save_name+'_LossPlot.png')
    
elif Action=='Test':
    LM=np.load('Loading_Mean.npy')
    LS=np.load('Loading_Stdev.npy')
    UM=np.load('Unloading_Mean.npy')
    US=np.load('Unloading_Stdev.npy')
    model_L = FCC_model()
    model_U = FCC_model()
    #X_Test=VAE_Encoder.predict(np.reshape(np.load('ML_Noise_Input_Files/ML_Noise_Input_Files.npy'),(100,20,60)),verbose=0)
    X_Test=np.load('ML_Noise_Input_Files/ML_Noise_Input_Files.npy')
    Y_Test_Load_Curve=(np.load('ML_Noise_Output_Files/ML_Noise_Output_Loading_Files.npy',allow_pickle=True)*LM)+LS
    Y_Test_Unload_Curve=(np.load('ML_Noise_Output_Files/ML_Noise_Output_Unloading_Files.npy',allow_pickle=True)*UM)+US
    model_L.load_weights('checkpoints/UC_NL_Load_Surrogate_Model_Weights/cp.ckpt')
    model_U.load_weights('checkpoints/UC_NL_Unload_Surrogate_Model_Weights/cp.ckpt')
    Tess=np.random.randint(0,100,10)
    Pred_Load=(model_L.predict(X_Test)*LM)+LS
    Pred_Unload=(model_U.predict(X_Test)*UM)+US
    for It in range(0,len(Tess)):
        plt.figure()
        
        plt.plot(np.arange(0,1.1,0.1),Y_Test_Load_Curve[Tess[It],:],'b-',label='True')
        plt.plot(np.arange(0,1.1,0.1),Pred_Load[Tess[It]],'r-',label='Pred')
        plt.plot(np.arange(1,-0.1,-0.1),Y_Test_Unload_Curve[Tess[It],:],'b-')
        plt.plot(np.arange(1,-0.1,-0.1),Pred_Unload[Tess[It]],'r-')
        plt.legend()
        
    #plt.scatter(Pred_Load,Y_Test_Load_Curve)
    #plt.xlabel('True Normalized Force Value')
    #plt.ylabel('Predicted Normalized Force Value')
    #plt.scatter(Pred_Unload,Y_Test_Unload_Curve)
    #plt.xlabel('True Normalized Force Value')
    #plt.ylabel('Predicted Normalized Force Value')

#plt.plot([0,1],[0,1],'r-')
elif Action=='Test_RNN':
    
    model_L=create_res_net()

    #X_Test=VAE_Encoder.predict(np.reshape(np.load('ML_Noise_Input_Files/ML_Noise_Input_Files.npy'),(100,20,60)),verbose=0)
    X_Test=np.load('ML_Noise_Input_Files/ML_RNN_Input_Files.npy')
    Y_Test_Load_Curve=np.load('ML_Noise_Output_Files/ML_RNN_Output_Energy_Files.npy',allow_pickle=True)
    model_L.load_weights('checkpoints/UC_NL_Energy_RNN_Model_Weights/cp.ckpt')
    Tess=np.random.randint(0,100,0)
    Pred_Load=model_L.predict(X_Test)

    for It in range(0,len(Tess)):
        plt.figure()
        
        plt.plot(np.arange(0,1.1,0.1),Y_Test_Load_Curve[Tess[It],:],'b-',label='True')
        plt.plot(np.arange(0,1.1,0.1),Pred_Load[Tess[It]],'r-',label='Pred')

        plt.legend()
        
    plt.scatter(Pred_Load,Y_Test_Load_Curve)
    plt.xlabel('True Normalized Force Value')
    plt.ylabel('Predicted Normalized Force Value')
    #plt.scatter(Pred_Unload,Y_Test_Unload_Curve)
    #plt.xlabel('True Normalized Force Value')
    #plt.ylabel('Predicted Normalized Force Value')

    plt.plot([0,1],[0,1],'r-')
elif Action=='Test_Coef':
    model_L = FCC_model()
    model_U = FCC_model()
    SML=np.load('Standardize_Mean_Loading.npy',allow_pickle=True)
    SSL=np.load('Standardize_Stdev_Loading.npy',allow_pickle=True)
    SMU=np.load('Standardize_Mean_Unloading.npy',allow_pickle=True)
    SSU=np.load('Standardize_Stdev_Unloading.npy',allow_pickle=True)
    X_Test=VAE_Encoder.predict(np.reshape(np.load('ML_Input_Files/Original_Inputs.npy'),(1000,20,60)),verbose=0)
    Y_Test_Load_Coef=np.load('ML_Output_Files/Original_Loading_Outputs.npy',allow_pickle=True)
    Y_Test_Unload_Coef=np.load('ML_Output_Files/Original_Unloading_Outputs.npy',allow_pickle=True)
    Y_Test_Load_XData=np.load('ML_Output_Files/Original_Loading_XData.npy',allow_pickle=True)
    Y_Test_Load_YData=np.load('ML_Output_Files/Original_Loading_YData.npy',allow_pickle=True)
    Y_Test_Unload_XData=np.load('ML_Output_Files/Original_Unloading_XData.npy',allow_pickle=True)
    Y_Test_Unload_YData=np.load('ML_Output_Files/Original_Unloading_YData.npy',allow_pickle=True)
    model_L.load_weights('checkpoints/UC_NL_Load_Surrogate_Model_Weights/cp.ckpt')
    model_U.load_weights('checkpoints/UC_NL_Unload_Surrogate_Model_Weights/cp.ckpt')
    Tess=[10,20,30,40]
    Pred_Load=(model_L.predict(X_Test)*SSL)+SML
    Pred_Unload=(model_U.predict(X_Test)*SSU)+SMU
    for It in range(0,len(Tess)):
        plt.figure()
        
        
        print('''------True Loading Coefficients----
{}
---------Predicted Loading Coefficients-----
{}
            
--------True Unloading Coefficients------
{}
-------Predicted Unloading Coefficients----
{}
              
              '''.format(Y_Test_Load_Coef[Tess[It]],Pred_Load[Tess[It]],Y_Test_Unload_Coef[Tess[It]],Pred_Unload[Tess[It]]))
        plt.plot(Y_Test_Load_XData[Tess[It]],Y_Test_Load_YData[Tess[It]],'b-',label='True')
        plt.plot(Y_Test_Load_XData[Tess[It]],func(Y_Test_Load_XData[Tess[It]],*Pred_Load[Tess[It]]),'r-',label='Pred')
        plt.plot(Y_Test_Unload_XData[Tess[It]],Y_Test_Unload_YData[Tess[It]],'b-',label='True')
        plt.plot(Y_Test_Unload_XData[Tess[It]],func(Y_Test_Unload_XData[Tess[It]],*Pred_Unload[Tess[It]]),'r-',label='Pred')
        plt.legend()

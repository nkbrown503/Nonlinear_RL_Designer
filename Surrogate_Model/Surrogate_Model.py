# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:39:22 2023

@author: nbrow
"""

from tensorflow.keras.layers import Input,Dense, Dropout,Attention,Flatten,Multiply,Add, BatchNormalization, ReLU
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.models import Model
import keras.backend as K
import numpy as np
from tensorflow.keras.losses import Huber

def FCC_model(args):
    inputs = Input(shape=(24,))
    t1= Dense(args.L1_nodes,activation='relu')(inputs)
    t1=BatchNormalization()(t1)

    
    for i in range(0,args.layers):
      t1=Dense(args.nodes,activation='relu')(t1)
      t1=BatchNormalization()(t1)
      t1=Dropout(args.dropout)(t1)
       
    t1=Dense(args.LF_nodes,activation='relu')(t1)
    t1=BatchNormalization()(t1)
        
    
    outputs1 = Dense(11)(t1)
      

    lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=5e-3,
    decay_steps=20000,
    decay_rate=0.9)
    
    model = Model(inputs, outputs1)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='mean_squared_error')


    return model
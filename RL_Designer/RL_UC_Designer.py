# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:18:38 2022

@author: nbrow
"""

import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import copy 
from UC_Env import UC_Env
import time
import matplotlib.pyplot as plt 
import gym
import argparse
import numpy as np
from DDPG import Agent
from utils import plot_learning_curve
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",type=float,default=2.5e-5)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--max_noise",type=float,default=0.3)
    parser.add_argument("--noise_decay",type=float,default=7.5e-6)
    parser.add_argument("--Trial_Num",type=int,default=2)
    
    args=parser.parse_args()
    return args

args=parse_args()           
    
n_games = 20000 #Number of training Episodes

figure_file = 'plots/UC_RL_Training.png'
 
best_score = -10
score_history = []


Test=True #If Test=False then the RL agent will be trained from scratch
env = UC_Env() #Call the RL environment 
agent = Agent(input_dims=(35,),alpha=args.learning_rate,beta=args.learning_rate,gamma=args.gamma,
              env=env,Start_Noise=args.max_noise,Noise_Decay=args.noise_decay,n_actions=7 ) #Call the RL agent 

if Test==False:
    
    for i in range(n_games):
        
        observation = env.reset(Test,i=0)
        done = False
        evaluate=False
        score = 0
        steps=0
        while not done:
            steps+=1 
            action = agent.choose_action(observation, evaluate)

            observation_, reward, done, Legal = env.step(action)                
            if (env.state_UC==env.state_UC_).all()==True and done==False and Legal:
                #Check if the agent tried to take the same action back to back 
                done=True
                
                
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation=observation_
        S2=time.time()
        if agent.memory.mem_cntr>300:
            #Introduce a slight delay in the learning for the agent 
            agent.learn()


        

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score and i>20:
            best_score = avg_score
            Saved=False
            while Saved==False:
                try:
                    agent.save_models()
                    Saved=True
                except:
                    time.sleep(0.1)
        #wandb.log({'best score': np.round(best_score,3),'reward': np.round(score,4),'avg reward': np.round(avg_score,4),'episode': i, 'noise': agent.noise})

        print('episode', i, 'score %.3f' % score, 'avg score %.3f' % avg_score, 'Steps %.0f' % steps, 'Noise %.4f' %agent.noise, ' LR %.3f' %agent.tau)

    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
else:
    a=time.time()
    Testing_Trials=1  #How many trials would you like to test?
    n_steps = 0
    Avg_reward=[]
    MAE=[]
    agent.load_models()
    DNL=[]
    fig2,ax2=plt.subplots()
    for i in range(Testing_Trials):
        
        Best_Reward=-1
        observation=env.reset(Test,i)
        done = False
        evaluate=True
        score = 0
        LR=-1
 
        while not done:
            action = agent.choose_action(observation, evaluate)
           
                
            observation_, reward, done, Legal = env.step(action)
            if env.step_count==3:
                done=True
            if reward>LR or reward==-1 or reward<LR:
                score += reward
                print(reward)
                print(env.Energy)


                if reward!=0 and reward!=-1:
                    LR=np.max([-env.Perc_Error,-env.Perc_Error2])
                    FE=env.Force_Error
                observation = observation_
     

            else:
                
                env.state_UC=env.state_UC_
                try:
                    env.Current_Force=env.Current_Force_
                except:
                    'Nothing'
                done=True
                
            if done or not done:
                
               
                env.render(Legal,i,ax2)
                

        

        print('Average Final Error: {}'.format(np.mean(Avg_reward)))

    
        
    
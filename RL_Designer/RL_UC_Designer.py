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
    #Select the hyperparameters 
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--max_noise",type=float,default=0.25)
    parser.add_argument("--noise_decay",type=float,default=4.5e-6)
    parser.add_argument("--Trial_Name",type=str,default='Generalized_NL_RL')
    
    args=parser.parse_args()
    return args

args=parse_args()           
    
n_games = 20_000 #Number of training Episodes

figure_file = 'plots/UC_RL_Training.png'
 
#Start with a low score to save over 
best_score = -10
score_history = []



Test=True #If Test=False then the RL agent will be trained from scratch
env = UC_Env() #Call the RL environment 

#call the DRL agent 
agent = Agent(input_dims=(28,),alpha=args.learning_rate,beta=args.learning_rate,gamma=args.gamma,
              env=env,Start_Noise=args.max_noise,Noise_Decay=args.noise_decay,n_actions=7 ) #Call the RL agent 

if Test==False: #Train the agent 
    wandb.init(project='RL_NL_UC_Training',
           config=vars(args),
           name='{}'.format(args.Trial_Name))
    for i in range(n_games):
        
        #Reset observation to fully voided unit cell and select new desired response
        observation = env.reset(Test,i=0)
        done = False
        evaluate=False
        score = 0
        steps=0
        while not done:
            steps+=1 
            
            #Given an observation, the agent should select an action
            action = agent.choose_action(observation, evaluate)
            
            #Take the action resulting in a new observation
            observation_, reward, done, Legal = env.step(action)                
            
            #Add the reward for the action to the total score of the episode
            score += reward
            
            #Save everything in the replay buffer 
            agent.remember(observation, action, reward, observation_, done)
            
            #New state becomes current state 
            observation=observation_
        S2=time.time()
        if agent.memory.mem_cntr>300: #Wait 300 steps before we start updating the weights of the agent 
            #Introduce a slight delay in the learning for the agent 
            agent.learn()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score and i>20: #Save the agent weights if it produces a new best average score 
            best_score = avg_score
            Saved=False
            while Saved==False:
                try:
                    agent.save_models()
                    Saved=True
                except:
                    time.sleep(0.1)
        wandb.log({'best score': np.round(best_score,3),'reward': np.round(score,4),'avg reward': np.round(avg_score,4),'episode': i, 'noise': agent.noise})

        print('episode', i, 'score %.3f' % score, 'avg score %.3f' % avg_score, 'Steps %.0f' % steps, 'Noise %.4f' %agent.noise, ' LR %.3f' %agent.tau)

    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
else:
    a=time.time()
    Testing_Trials=5  #How many trials would you like to test?
    n_steps = 0
    #Load the weights of the trained agent 
    agent.load_models()
    DNL=[]
    Good_Design=np.zeros((Testing_Trials,))
    fig2,ax2=plt.subplots()

    for i in range(Testing_Trials):
        
        Best_Reward=-1
        
        #Void the design and select a desired response, the user can uncomment the line two below to select max or min hysteresis 
        observation=env.reset(Test,i)
        #env.obs[0]=int(input('Would you like to design an energy absorbing (Enter 0) or energy returning material (Enter 1)?:  '))
        env.obs[0]=0
        done = False
        evaluate=True
        score = 0
        LR=-1
 	
        while not done:
            action = agent.choose_action(observation, evaluate)
            
            observation_, reward, done, Legal = env.step(action)
            if (env.state_UC==env.state_UC_).all()==True and done==False and Legal:
                #Check if the agent tried to take the same action back to back 
                done=True
            
            if Legal and done:
                print('Design Step: {}   Reward: {}   Curve_Error: {}   Energy_Return:{} \n'.format(env.step_count,np.round(reward,4),np.round(env.Loading_Error,4),np.round(env.Energy_Return,4)))

            if reward>LR or reward==-1 or reward<LR:
                score += reward
                
                if reward!=0 and reward!=-1:
                    LR=np.max([-env.Loading_Error])
                    
                observation = observation_
            else:
                
                env.state_UC=env.state_UC_
                try:
                    env.Current_Force=env.Current_Force_
                except:
                    'Nothing'
                done=True
               
            env.render(Legal,i,ax2,Turn=0)
            
        Best_Reward=-1
        #np.save('Energy_Absorb_Design.npy',env.state_UC)
        observation=env.reset(Test,i)
        
        #env.obs[0]=int(input('Would you like to design an energy absorbing (Enter 0) or energy returning material (Enter 1)?:  '))
        env.obs[0]=1
        done = False
        evaluate=True
        score = 0
        LR=-1
     	
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, Legal = env.step(action)
            if (env.state_UC==env.state_UC_).all()==True and done==False and Legal:
                #Check if the agent tried to take the same action back to back 
                done=True
            
            if Legal and done:
                print('Design Step: {}   Reward: {}   Curve_Error: {}   Energy_Return:{} \n'.format(env.step_count,np.round(reward,4),np.round(env.Loading_Error,4),np.round(env.Energy_Return,4)))
            #else:
            #    print('Design Step: {}   Reward: {}   Curve_Error: {}   Energy_Return:{} \n'.format(env.step_count,np.round(reward,4),'N/A','N/A'))

            if reward>LR or reward==-1 or reward<LR:
                score += reward
                
                if reward!=0 and reward!=-1:
                    LR=np.max([-env.Loading_Error])
                    
                observation = observation_
            else:
                
                env.state_UC=env.state_UC_
                try:
                    env.Current_Force=env.Current_Force_
                except:
                    'Nothing'
                done=True
               
            env.render(Legal,i,ax2,Turn=1)

    
        
    

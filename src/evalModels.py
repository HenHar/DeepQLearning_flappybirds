from qlearn import buildmodel, imgTransform

import os 
os.environ['SDL_VIDEODRIVER']='dummy'
import pygame 
import pygame.transform
import pygame.display
import sys 

import matplotlib.pyplot as plt
import pickle
import pylab
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
import skimage as skimage
import tensorflow as tf
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input

ACTIONS = 2 # number of valid actions
MAX_GAMES = 10
MAX_SCORE = 1000
LEARNING_RATE = 10e-4

PIPE_GAP_SIZE = 150
REWARD_ALIVE = True
EVAL_PATH = "evalModels_hard2_Ra.pickle"
MODEL_PATH = "model_hard2_Ra_"

def evalModel(model):
    evalModels = []
    model_ids = list(range(10000, 510000, 10000))

    for i, m_id in enumerate(model_ids):
        evalModels.append([])
        # open up a game state to communicate with emulator
        game_state = game.GameState(PIPE_GAP_SIZE, REWARD_ALIVE)
        # store the previous observations in replay memory
        D = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t_colored, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = imgTransform(x_t_colored)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        #In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
 
        print ("Now we load weight")
        model.load_weights(MODEL_PATH + str(m_id) + ".h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)    
        num_games = 0
        score = 0
        t = 0
        while (num_games < MAX_GAMES):
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([ACTIONS])

            q = model.predict(s_t)       #input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[max_Q] = 1

            #run the selected action and observed next state and reward
            x_t1_colored, reward, terminal = game_state.frame_step(a_t)

            if reward == -1:
                print("new game, score: ", score)
                evalModels[i].append(score)
                score = 0
                num_games += 1
                
            if reward == 1:
                score += 1
                if score == MAX_SCORE:
                    print("new game, max score reached: ", score)
                    evalModels[i].append(score)
                    score = 0
                    num_games += 1
                    game_state.reset()
                    
            x_t1 = imgTransform(x_t1_colored)
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            s_t = s_t1
            t = t + 1
        print("scores: ", evalModels)
    with open(EVAL_PATH, 'wb') as handle:
        pickle.dump(evalModels, handle, protocol=pickle.HIGHEST_PROTOCOL)


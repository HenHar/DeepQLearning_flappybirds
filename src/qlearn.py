import os 
#os.environ['SDL_VIDEODRIVER']='dummy'
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

PIPE_GAP_SIZE = 100
REWARD_ALIVE = True
TRAIN_EVAL_PATH = "trainEval_hard_Ra_fixedTarget.pickle"
MODEL_NAME = "model_hard_Ra_fixedTarget"

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.95 # decay rate of past observations
OBSERVATION = 1000. # timesteps to observe before training
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.2 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
MAX_ITER = 500000
EXPLORE = MAX_ITER - 100000 # frames over which to anneal epsilon
LOG_STATES = False
C_TARGET = 1000



img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def imgTransform(img_colored):
    img = skimage.color.rgb2gray(img_colored)
    img = skimage.transform.resize(img,(80,80))
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
    img = img / 255.0
    return img

def buildmodel():
    print("Now we build the model", flush = True)
    model = Sequential()
    #model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    #model.add(Activation('relu'))
    #model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    #model.add(Activation('relu'))
    #model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    #model.add(Activation('relu'))
    #model.add(Flatten())
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    #model.add(Dense(2))

    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation = "relu", padding='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation = "relu", padding='same'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation = "relu", padding='same'))
    model.add(Flatten())
    model.add(Dense(512, activation = "relu", ))
    model.add(Dense(2))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model", flush = True)
    return model
    
def trainNetwork(model, target = True):
    trainEval =  {"loss" : [], "q_max": [], "q_mean": [], "score": [], "time_step_score": [], "epsilon": []}
    
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
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    if target:
        # init target network
        model_target= tf.keras.models.clone_model(model)
        model_target.set_weights(model.get_weights())

    score = 0
    t = 0
    while (t < MAX_ITER):

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                #print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, reward, terminal = game_state.frame_step(a_t)

        #evaluation of scores during training
        if reward == -1:
            trainEval["score"].append(score)
            trainEval["time_step_score"].append(t)
            score = 0
        else:
            score += reward

        x_t1 = imgTransform(x_t1_colored)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, reward, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            #Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            if target:
                targets = model_target.predict(state_t)
            else:
                targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

            loss += model.train_on_batch(state_t, targets)

        s_t = s_t1
        t = t + 1
        
        # update target network weights
        if target and t % C_TARGET == 0:
            model_target.set_weights(model.get_weights())

        if t % 10000 == 0:
            print("TIMESTEP", t, "save model and evaluation parameter", flush = True)
            model_path = MODEL_NAME + str(t) + ".h5"
            model.save_weights(model_path, overwrite=True),
            with open(TRAIN_EVAL_PATH, 'wb') as handle:
                pickle.dump(trainEval, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % 1  == 0 and LOG_STATES:
            print("TIMESTEP", t, "/ STATE", state, \
               "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss, flush = True)
        trainEval["loss"].append(loss)
        trainEval["q_max"].append(np.max(Q_sa))
        trainEval["q_mean"].append(np.mean(Q_sa))
        trainEval["epsilon"].append(epsilon)

    print("Training finished!")
    print("************************")


    



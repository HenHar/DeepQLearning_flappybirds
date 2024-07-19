import pickle
import matplotlib.pyplot as plt
import numpy as np

def getImages():
    game_state = game.GameState(150)
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t_colored, r_0, terminal = game_state.frame_step(do_nothing)
    img = np.rot90(x_t_colored, 3)
    plt.imshow(img)
    plt.show()
    img = skimage.color.rgb2gray(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    img = skimage.transform.resize(img,(80,80))
    plt.imshow(img, cmap='gray')
    plt.show()
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
    img = img / 255.0
    plt.imshow(img, cmap='gray')
    plt.show()

def evalTraining(path):
    with open(path, 'rb') as handle:
        trainEval = pickle.load(handle)
    losses = trainEval["loss"]
    q_max = trainEval["q_max"]
    q_mean = trainEval["q_mean"]
    score = trainEval["score"]
    time_steps_score = trainEval["time_step_score"]
    epsilon = trainEval["epsilon"]

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    time_steps = range(len(losses))
    game_steps = range(len(score))
    ax1.plot(time_steps_score, score, 'tab:orange')
    ax1.set_xlabel("time steps [frames]")
    ax1.set_ylabel("score")

    ax2.plot(time_steps, losses, 'tab:green')
    ax2.ticklabel_format(style='sci',axis='x', scilimits=(0,0), useMathText = True)
    ax2.set_xlabel("time steps [frames]")
    ax2.set_ylabel("loss")

    ax3.plot(time_steps, q_max, 'tab:red', label = "q_max")
    ax3.plot(time_steps, q_mean, 'tab:blue', label = "q_mean")
    ax3.ticklabel_format(style='sci',axis='x', scilimits=(0,0), useMathText = True)
    ax3.set_xlabel("time steps [frames]")
    ax3.set_ylabel("value")
    ax3.legend()


    fig, ax = plt.subplots()
    ax.plot(time_steps, epsilon)
    ax.set_xlabel("time steps [frames]")
    ax.set_ylabel("epsilon value")
    ax.ticklabel_format(style='sci',axis='x', scilimits=(0,0), useMathText = True)
 

    plt.show()

def evalScores(path):
    with open(path, 'rb') as handle:
        eval_results = pickle.load(handle)
    scores_mean = []
    scores_std = []
    for s in eval_results:
        scores_mean.append(np.mean(s))
        scores_std.append(np.std(s))
        print("mean: ", np.mean(s), " std: ", np.std(s))

    time_steps = range(10000, 510000, 10000)
    fig, ax = plt.subplots()
    fig.suptitle('Mean score every 10000 steps', fontsize=12)
    ax.plot(time_steps, scores_mean)
    ax.set_xlabel("time steps [frames]")
    ax.set_ylabel("mean score (10 games)")
    plt.show()
    
       
path = "...."    
evalScores(path)
 
path = "...."    
evalTraining(path)



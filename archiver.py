'''
█████╗ ██████╗  ██████╗██╗  ██╗██╗██╗   ██╗███████╗██████╗ 
██╔══██╗██╔══██╗██╔════╝██║  ██║██║██║   ██║██╔════╝██╔══██╗
███████║██████╔╝██║     ███████║██║██║   ██║█████╗  ██████╔╝
██╔══██║██╔══██╗██║     ██╔══██║██║╚██╗ ██╔╝██╔══╝  ██╔══██╗
██║  ██║██║  ██║╚██████╗██║  ██║██║ ╚████╔╝ ███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝
'''

import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import json
import shutil

# The above class, MyClass, is used to store and manipulate lists of rewards, epsilon values, step
# counts, and fruit counters, and provides methods to calculate averages and save a figure of the
# data.
class Archiver():
    
    def __init__(self, every, name, num_experiments):
        """
        The above function is the initialization method for a class and it initializes several instance
        variables.
        """

        # The code `if not os.path.isdir('plots'): os.makedirs('plots')` is checking if a directory named
        # "plots" exists in the current working directory. If the directory does not exist, it creates a new
        # directory named "plots". This is done to ensure that the directory exists before saving any plots to
        # it.
        if not os.path.isdir("Experiments"):
            os.makedirs("Experiments")

        self.timestamp = date_time = datetime.fromtimestamp(time.time())
        self.timestring = date_time.strftime("%m_%d_%Y_%H_%M_%S")
        self.NAME = name
        self.experimentRoot = "Experiments/" + self.NAME + "_" + self.timestring
        if not os.path.isdir(self.experimentRoot):
            os.makedirs(self.experimentRoot)
            os.makedirs(self.experimentRoot + '/plots')
            os.makedirs(self.experimentRoot + '/models')
            os.makedirs(self.experimentRoot + '/data')
            os.makedirs(self.experimentRoot + '/code')

        self.ep_rewards = []
        self.ep_rewards_norm = []
        self.epsilon_over_time = []
        self.steps_before_death = []
        self.fruits_eaten = []
        self.AGGREGATE_STATS_EVERY = every
        self.average_reward_list = []
        self.average_step_list = []
        self.average_fruit_list = []
        self.average_epsilon_list = []
        self.min_reward_list = []
        self.max_reward_list = []
        self.num_experiments = num_experiments
        self.cpu = []
        self.average_cpu = []
        self.ram = []
        self.average_ram = []
        self.gpu_load = []
        self.average_gpu_load = []
        self.gpu_mem = []
        self.average_gpu_mem = []
        self.step_time = []
        self.average_step_time = []
        self.timestamp = date_time = datetime.fromtimestamp(time.time())


    def appendLists(self, episode_reward, epsilon, step_count, fruit_counter, cpu, ram, step_time, gpu_load, gpu_mem):
        """
        The function appends various values to different lists.
        
        :param episode_reward: The total reward obtained in an episode of the game
        :param epsilon: Epsilon is a parameter used in reinforcement learning algorithms, specifically in
        the context of exploration vs exploitation trade-off. It determines the probability of choosing a
        random action instead of the action with the highest expected reward
        :param step_count: The number of steps taken in the episode before the agent died or the episode
        ended
        :param fruit_counter: The `fruit_counter` parameter keeps track of the number of fruits eaten in
        each episode
        """
        self.ep_rewards.append(episode_reward)
        self.ep_rewards_norm.append(episode_reward/100)
        self.epsilon_over_time.append(epsilon)
        self.steps_before_death.append(step_count)
        self.fruits_eaten.append(fruit_counter)
        self.cpu.append(cpu)
        self.ram.append(ram)
        self.gpu_load.append(gpu_load)
        self.gpu_mem.append(gpu_mem)
        self.step_time.append(step_time)

    def averageLists(self):
        """
        The function calculates the average, minimum, and maximum values of rewards, steps, and fruits eaten
        over a specified number of episodes and appends them to respective lists.
        """
        average_reward = sum(self.ep_rewards[-self.AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-self.AGGREGATE_STATS_EVERY:])
        average_step = sum(self.steps_before_death[-self.AGGREGATE_STATS_EVERY:])/len(self.steps_before_death[-self.AGGREGATE_STATS_EVERY:])
        average_fruits = sum(self.fruits_eaten[-self.AGGREGATE_STATS_EVERY:])/len(self.fruits_eaten[-self.AGGREGATE_STATS_EVERY:])
        min_reward = min(self.ep_rewards[-self.AGGREGATE_STATS_EVERY:])
        max_reward = max(self.ep_rewards[-self.AGGREGATE_STATS_EVERY:])
        average_cpu = sum(self.cpu[-self.AGGREGATE_STATS_EVERY:])/len(self.cpu[-self.AGGREGATE_STATS_EVERY:])
        average_ram = sum(self.ram[-self.AGGREGATE_STATS_EVERY:])/len(self.ram[-self.AGGREGATE_STATS_EVERY:])
        average_gpu_load = sum(self.gpu_load[-self.AGGREGATE_STATS_EVERY:])/len(self.gpu_load[-self.AGGREGATE_STATS_EVERY:])
        average_gpu_mem = sum(self.gpu_mem[-self.AGGREGATE_STATS_EVERY:])/len(self.gpu_mem[-self.AGGREGATE_STATS_EVERY:])
        average_step_time = sum(self.step_time[-self.AGGREGATE_STATS_EVERY:])/len(self.step_time[-self.AGGREGATE_STATS_EVERY:])
        average_epsilon = sum(self.epsilon_over_time[-self.AGGREGATE_STATS_EVERY:])/len(self.epsilon_over_time[-self.AGGREGATE_STATS_EVERY:])

        self.average_reward_list.append(average_reward)
        self.average_step_list.append(average_step)
        self.average_fruit_list.append(average_fruits)
        self.min_reward_list.append(min_reward)
        self.max_reward_list.append(max_reward)
        self.average_cpu.append(average_cpu)
        self.average_ram.append(average_ram)
        self.average_gpu_load.append(average_gpu_load)
        self.average_gpu_mem.append(average_gpu_mem)
        self.average_step_time.append(average_step_time)
        self.average_epsilon_list.append(average_epsilon)
        
        del average_reward, average_step, average_fruits, min_reward, max_reward, average_cpu, average_ram, average_step_time, average_gpu_load, average_gpu_mem, average_epsilon

    def saveSetup(self,ACTION_SPACE_SIZE, WIDTH, HEIGHT, START_LENGTH, NUM_FRUIT, CAN_PORT, EPISODES, DISCOUNT, REPLAY_MEMORY_SIZE, 
                  MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, AGGREGATE_STATS_EVERY, LOG_EVERY_STEP, EXPERIMENT_NAME, 
                  MAX_STEPS, reward_fruit, reward_into_self, reward_step, reward_wall, epsilon, EPSILON_DECAY, MIN_EPSILON, EPISODES_BEFORE_DECAY, model, summary_string,
                  renderVisual, renderText, renderText_conv, renderText_num, sleepText, sleepVisual, RENDER_EVERY, mode, useRGBinput, stateDepth, 
                  trackGPU, trackCPU_RAM, GPU_id, spawnDistanceFromWall, imageResizeFactor, input_dims, good_mem_size_muliplier, good_mem_min_multiplier, good_mem_split, good_mem_threshold, use_good_mem, reward_distance_exponent, useDifferentColorHead, noNegRewards, notes):
        '''
        The `saveSetup` function saves the setup and hyperparameters of a model to a file.
        '''
        # saving setup 
        json_model = json.dumps(model, indent = 4)  

        setup = f"{ACTION_SPACE_SIZE=}\n{WIDTH=}\n{HEIGHT=}\n{START_LENGTH=}\n{NUM_FRUIT=}\n{CAN_PORT=}\n{EPISODES=}\n{DISCOUNT=}\n{REPLAY_MEMORY_SIZE=}\n"
        setup += f"{MIN_REPLAY_MEMORY_SIZE=}\n{MIN_REPLAY_MEMORY_SIZE=}\n{MINIBATCH_SIZE=}\n{UPDATE_TARGET_EVERY=}\n{epsilon=}\n{AGGREGATE_STATS_EVERY=}\n{LOG_EVERY_STEP=}\n{EXPERIMENT_NAME=}\n"
        setup += f"{MAX_STEPS=}\n{reward_fruit=}\n{reward_into_self=}\n{reward_step=}\n{reward_wall=}\n{epsilon=}\n{EPSILON_DECAY=}\n"
        setup += f"{MIN_EPSILON=}\n{EPISODES_BEFORE_DECAY=}\n{renderVisual=}\n{renderText=}\n{renderText_conv=}\n{renderText_num=}\n{sleepText=}\n{sleepVisual=}\n{RENDER_EVERY=}\n"
        setup += f"{mode=}\n{useRGBinput=}\n{stateDepth=}\n{trackGPU=}\n{trackCPU_RAM=}\n{GPU_id=}\n{spawnDistanceFromWall=}\n{imageResizeFactor=}\n"
        setup += f"{good_mem_size_muliplier=}\n{good_mem_min_multiplier=}\n{good_mem_split=}\n{good_mem_threshold=}\n{use_good_mem=}\n{reward_distance_exponent=}\n{useDifferentColorHead=}\n{noNegRewards=}\n{notes=}\n"
        setup += "\nModel Summary=\n" + f"{input_dims=}\n" +  summary_string + "\n=====MORE DETAILED VERSION BELOW=====\nDetailed Model=\n" + json_model
        print(setup, file=open(self.experimentRoot + "/setup.out", 'w'))  # saves hyperparameters to the experiment folder
        del json_model

        for filename in os.listdir("./"): # save code base to replicate further if needed
            print(filename)
            if filename.endswith(".py"):
                shutil.copyfile(filename, self.experimentRoot + "/code/" + filename)
        

    def saveModel(self, model):
        """
        The `saveModel` function saves a given tf model to a specific file path.
        
        :param model: The `model` parameter is an instance of a machine learning model that you want to save
        """
        model.save(self.experimentRoot + "/models/" + self.NAME + str(len(self.average_reward_list)) + ".keras")

    def saveData(self):
        """
        The `saveData` function saves the data from various lists into a numpy file.
        """
        movingData = {
            "average_reward_list": self.average_reward_list,
            "average_step_list": self.average_step_list,
            "average_fruit_list": self.average_fruit_list,
            "min_reward_list": self.min_reward_list,
            "max_reward_list": self.max_reward_list,
            "average_cpu": self.average_cpu,
            "average_ram": self.average_ram,
            "average_gpu_load": self.average_gpu_load,
            "average_gpu_mem": self.average_gpu_mem,
            "average_step_time": self.average_step_time,
            "average_epsilon_list": self.average_epsilon_list
            }

        np.save(self.experimentRoot + "/data/" + self.NAME + str(len(self.average_reward_list)) + ".npy", movingData)
        del movingData

    def saveFig(self):
        """
        The `saveFig` function saves a figure with multiple plots to a file.
        """
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward/Steps')

        
        ax1.plot(self.average_reward_list, label = "average Reward")
        ax1.plot([100 * x for  x in self.average_epsilon_list], label = "epsilon percentage", color="#6B6B6B")
        ax1.plot(self.min_reward_list, label = "min. Reward")
        ax1.plot(self.max_reward_list, label = "max. Reward")
        ax1.plot(self.average_step_list, label = "average Steps")
        plt.legend(loc="upper left")
        plt.ylim([-200, 500])

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('Fruits Eaten')  # we already handled the x-label with ax1
        ax2.plot(self.average_fruit_list, label = "Average Fruits eaten", color="#976FE8")
        ax2.tick_params(axis='y')
        plt.ylim([0, 10])
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.legend(loc="upper right")
        plt.savefig(self.experimentRoot + "/plots/" + self.NAME + str(len(self.average_reward_list)) + ".png")
        plt.clf()

        del fig, ax1, ax2
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
import os
import numpy as np

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
        if not os.path.isdir('plots'):
            os.makedirs('plots')

        self.ep_rewards = []
        self.ep_rewards_norm = []
        self.epsilon_over_time = []
        self.steps_before_death = []
        self.fruits_eaten = []
        self.AGGREGATE_STATS_EVERY = every
        self.NAME = name
        self.average_reward_list = []
        self.average_step_list = []
        self.average_fruit_list = []
        self.min_reward_list = []
        self.max_reward_list = []
        self.num_experiments = num_experiments

    def appendLists(self, episode_reward, epsilon, step_count, fruit_counter):
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
        self.average_reward_list.append(average_reward)
        self.average_step_list.append(average_step)
        self.average_fruit_list.append(average_fruits)
        self.min_reward_list.append(min_reward)
        self.max_reward_list.append(max_reward)
        del average_reward, average_step, average_fruits, min_reward, max_reward
        
        
    def saveFig(self):
        """
        The `saveFig` function saves a figure with multiple plots to a file.
        """
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward/Steps')

        
        ax1.plot(self.average_reward_list, label = "average Reward")
        # plt.plot(epsilon_over_time * 100, label = "epsilon percentage")
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
        plt.xlim([0, self.num_experiments])
        # plt.xticks(np.arange(0, self.num_experiments/100, self.AGGREGATE_STATS_EVERY))

        plt.legend(loc="upper right")
        plt.savefig("./plots/" + self.NAME + '-avg-wall-' + str(len(self.average_reward_list)) + '-' + str(time.time()) + ".png")
        plt.clf()

        del fig, ax1, ax2
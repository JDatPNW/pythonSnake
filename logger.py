'''
██╗      ██████╗  ██████╗  ██████╗ ███████╗██████╗ 
██║     ██╔═══██╗██╔════╝ ██╔════╝ ██╔════╝██╔══██╗
██║     ██║   ██║██║  ███╗██║  ███╗█████╗  ██████╔╝
██║     ██║   ██║██║   ██║██║   ██║██╔══╝  ██╔══██╗
███████╗╚██████╔╝╚██████╔╝╚██████╔╝███████╗██║  ██║
╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
                                                   
'''

import os
from colorama import Fore, Style

# The Logger class is used to log information about each episode and step of a program.
class Logger():
    def __init__(self):
        """
        The above function is an empty constructor for a Python class.
        """
        pass

    def log(self, episode, step, reward, reward_overall, action, direction, head, fruit, dead, epsilon, ran_into_self, cause, eaten, distance, cpu, ram, step_time, gpu_load, gpu_mem, gpu_id):
        """
        The `log` function prints a formatted log message with various information about the episode, step,
        rewards, actions, and other details.
        
        :param episode: The episode number of the log entry
        :param step: The step parameter represents the current step or iteration in the training process
        :param reward: The reward parameter represents the reward received at a particular step in the
        episode
        :param reward_overall: The overall reward accumulated in the episode
        :param action: The action taken by the agent in the current step
        :param direction: The "direction" parameter represents the current direction of the snake's head. It
        can be one of the following values: "up", "down", "left", or "right"
        :param head: The "head" parameter refers to the current position of the snake's head in the game
        :param dead: The "dead" parameter indicates whether the agent is dead or alive at the current step.
        It is a boolean value, where True indicates that the agent is dead and False indicates that the
        agent is alive
        :param epsilon: The parameter "epsilon" represents the exploration rate of the agent. It determines
        the probability of the agent choosing a random action instead of the optimal action based on its
        current policy
        :param ran_into_self: The parameter "ran_into_self" is a boolean value that indicates whether the
        snake ran into itself during the current step
        :param cause: The "cause" parameter in the log function is used to indicate the cause of the snake's
        death. It provides information about why the snake died, such as hitting a wall, colliding with
        itself, or reaching a maximum number of steps
        """
        print("\n\n")
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(Fore.RED)
        print ('| {:<5} | {:<5} | {:<13} | {:<12} | {:<15} | {:<15} | {:<7} | {:<10} | {:<10} | {:<10} | {:<10} |'.format('EP','Step', 'Epsilon', 'Step Reward', 'Fruits Eaten', 'Ep Reward', 'Action', 'Dir', 'Head', 'Fruit', 'Distance'))
        print(Style.RESET_ALL)
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(Fore.YELLOW)
        print ('| {:<5} | {:<5} | {:<13} | {:<12} | {:<15} | {:<15} | {:<7} | {:<10} | {:<10} | {:<10} | {:<10} |'.format(episode, step, round(epsilon, 10), round(reward, 4), eaten, round(reward_overall, 1), action, str(direction), str(head), str(fruit), round(distance, 5)))
        print(Style.RESET_ALL)
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(Fore.RED)
        print ('| {:<7} | {:<12} | {:<12} | {:<7} | {:<7} | {:<7} | {:<10} | {:<10} | {:<10} | {:<10} |'.format('Dead', 'Cause', 'Into Self', 'PID', 'CPU %', 'RAM %', 'GPU ID', 'GPU Load %', 'GPU Mem %', 'Step Time'))
        print(Style.RESET_ALL)
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(Fore.YELLOW)
        print ('| {:<7} | {:<12} | {:<12} | {:<7} | {:<7} | {:<7} | {:<10} | {:<10} | {:<10} | {:<10} |'.format(dead, cause, ran_into_self, os.getpid(), round(cpu, 5), round(ram, 5), gpu_id, round(gpu_load, 5), round(gpu_mem, 5), round(step_time, 7)))
        print(Style.RESET_ALL)
        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        print("\n\n")
        print(Fore.GREEN)
        print("""
                    /^\/^\           ██╗    ██╗ █████╗ ██╗████████╗██╗███╗   ██╗ ██████╗             . ~ ------- ~ .
                  __|_| O|  \        ██║    ██║██╔══██╗██║╚══██╔══╝██║████╗  ██║██╔════╝           .'  .~ _____ ~-. `.
           \/    /~    \_/    \      ██║ █╗ ██║██   ██║██║   ██║   ██║██╔██╗ ██║██║  ███╗         /   /             `.`.
            \____|__________/  \     ██║ █╗ ██║███████║██║   ██║   ██║██╔██╗ ██║██║  ███╗        /   /                `'
                    \_______      \  ██║███╗██║██╔══██║██║   ██║   ██║██║╚██╗██║██║   ██║       /    |
                            \        ╚███╔███╔╝██║  ██║██║   ██║   ██║██║ ╚████║╚██████╔╝██╗██╗██╗  /
                              \       ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝╚═╝                      
              """)
        print(Style.RESET_ALL)
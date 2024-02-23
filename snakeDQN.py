'''
███╗   ███╗ █████╗ ██╗███╗   ██╗    ██╗      ██████╗  ██████╗ ██████╗ 
████╗ ████║██╔══██╗██║████╗  ██║    ██║     ██╔═══██╗██╔═══██╗██╔══██╗
██╔████╔██║███████║██║██╔██╗ ██║    ██║     ██║   ██║██║   ██║██████╔╝
██║╚██╔╝██║██╔══██║██║██║╚██╗██║    ██║     ██║   ██║██║   ██║██╔═══╝ 
██║ ╚═╝ ██║██║  ██║██║██║ ╚████║    ███████╗╚██████╔╝╚██████╔╝██║     
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     
                                                                      
'''                                                             
                                                                      
# library imports
import numpy as np
from tqdm import tqdm
import colored_traceback
colored_traceback.add_hook()
import tensorflow as tf
from memory_profiler import profile
import gc
import io
import math
import psutil
import time
import GPUtil

# class imports
import DQNAgent
import snakeGame
import archiver
import logger
import renderer

# NOTE NOTE NOTE TODO: make updates to all new functions to use parameters (no magic numbers) and to be tracked by archiver and logger

# These are the parameters and settings used in the snake game and the DQN agent. Here is a brief
# explanation of each parameter:
ACTION_SPACE_SIZE = 4 # Number possible actions
WIDTH = 12 # Width of playable field
HEIGHT = 12 # Height of playable field
START_LENGTH = 3 # Starting Length for snake
NUM_FRUIT = 1 # NUMBER OF APPLES SPAWNED
CAN_PORT = False # Can the snake come back from the opposite site when hitting the wall?
EPISODES = 50_000 # Number of episodes
DISCOUNT = 0.99 # Discount factor / alpha
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
AGGREGATE_STATS_EVERY = 100  # episodes used for averaging for plotting
LOG_EVERY_STEP = True # Log into console every step?
TF_VERBOSE = 0 # TF print outs
EXPERIMENT_NAME = "same_as_my_computer" # Name used for files and folders for data
MAX_STEPS = 150 # Steps before game will automatically reset (to keep the game of going on forever once the agent becomes very good at playing)

reward_fruit = 25 # reward for picking up a fruit
reward_into_self = 0 # reward for trying to run into oneself (180 turn)
reward_step = 0 # reward given at every step
reward_wall = -50 # reward for walking into the wall and dying
reward_distance = True # whether or not to use the distance reward (recommended to use)
reward_distance_exponent = 10 # The exponent of by which the distance reward will be calculated, the larger the number, the smaller the reward
RENDER_EVERY = 1 # every n-th episode the game will be rendered

epsilon = 1 # Start Value for Epsilon
EPSILON_DECAY = 0.9995 # Rate at which Epsilon decays
MIN_EPSILON = 0.001 # Value where that decay stops
EPISODES_BEFORE_DECAY = 31 # episodes before epsilon dacay will start

renderVisual = False # uses pygame to draw the game state
renderText = False # Uses print statements to print the game
renderText_conv = False # renders text and converts it for better readability
renderText_num = False # renders text and keeps number format - better for debugging
sleepText = 0 # time the game will sleep between text state print renders
sleepVisual = 0 # time the game will sleep between visual state print renders 
trackCPU_RAM = False # if you want to track CPU usage
trackGPU = False # can be used when using a GPU - WARNING very slow! Also does measure GPU usage of system! not only process
GPU_id = 0 # use the ID of the GPU that is being used

input_dims = [WIDTH, HEIGHT, 1] # for non RGB input
useRGBinput = False # use screenshot of the game as opposed to the minimal input
imageResizeFactor = 6 # Factor by which theoriginal RGB image will be shrunk
spawnDistanceFromWall = 3 # Distance with which the agent will at least spawn from wall
stateDepth = 1 # NOTE: make sure to set to 1 if not using!!. How many images should be stacked for the input? To portrait motion (only really meant for RGB, but should also work with minimal input)
useDifferentColorHead = True

good_mem_size_muliplier = 0.5
good_mem_min_multiplier = 0.33
good_mem_split = 0.5
good_mem_threshold = 0.05 
use_good_mem = True

mode = "RGB: " + str(useRGBinput) + ", Depth: " + str(stateDepth)

notes = "Changed snake to be all single color and using depth 2 " # Add notes here about the experiment that might be of use, will be saved to setup file


# The code is creating instances of three different classes: `DQNAgent`, `snakeGame`,, `Archiver` and 'Logger'.
render = renderer.Renderer(renderText, renderVisual, WIDTH, HEIGHT, renderText_conv, renderText_num, useRGBinput, imageResizeFactor, useDifferentColorHead)
game = snakeGame.snakeGame(WIDTH, HEIGHT, START_LENGTH, NUM_FRUIT, CAN_PORT, 
                           reward_step, reward_fruit, reward_into_self, reward_wall, spawnDistanceFromWall)
plot = archiver.Archiver(AGGREGATE_STATS_EVERY, EXPERIMENT_NAME, EPISODES)
log = logger.Logger()

if useRGBinput:
    input_dims = render.Screenshot_Size
    if stateDepth > 1:
        input_dims[:0] = [stateDepth]

agent = DQNAgent.DQNAgent(REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, 
                          DISCOUNT, WIDTH, HEIGHT, ACTION_SPACE_SIZE, TF_VERBOSE, input_dims, useRGBinput,
                          good_mem_size_muliplier, good_mem_min_multiplier, good_mem_split, good_mem_threshold, use_good_mem)

stream = io.StringIO()
agent.target_model.summary(print_fn=lambda x: stream.write(x + '\n'))
summary_string = stream.getvalue()
stream.close()
del stream

process = psutil.Process() # tracks cpu and ram
if trackGPU and agent.has_gpu: # checks if tf sees GPU, gets id and uses for tracking
    gpu = GPUtil.getGPUs()
    gpu = gpu[agent.gpu_id]

plot.saveSetup(ACTION_SPACE_SIZE, WIDTH, HEIGHT, START_LENGTH, NUM_FRUIT, CAN_PORT, EPISODES, DISCOUNT, 
               REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, AGGREGATE_STATS_EVERY, 
               LOG_EVERY_STEP, EXPERIMENT_NAME, MAX_STEPS, reward_fruit, reward_into_self, reward_step, reward_wall, epsilon, 
               EPSILON_DECAY, MIN_EPSILON, EPISODES_BEFORE_DECAY, agent.model.get_config(), summary_string, 
               renderVisual, renderText, renderText_conv, renderText_num, sleepText, sleepVisual, RENDER_EVERY, mode, useRGBinput, stateDepth,
               trackGPU, trackCPU_RAM, GPU_id, spawnDistanceFromWall, imageResizeFactor, input_dims,
               good_mem_size_muliplier, good_mem_min_multiplier, good_mem_split, good_mem_threshold, use_good_mem, notes)

def main(episode):
    """
    The main function is a loop that runs episodes of a game, where each episode consists of multiple
    steps.
    """
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step_count = 1
    fruit_counter = 0
    run_into_self = False
    # Reset environment and get initial state
    dead = False
    deep_state = []
    state_ready = False
    prev_dist = 0

    if useRGBinput:
        render.InitPygame()
    elif not episode % RENDER_EVERY:
        if renderVisual:
            render.InitPygame()

    current_state = np.array(list(game.initGame()))
    if useRGBinput:
        current_state = np.array(render.getScreenshot(current_state))
        if stateDepth > 1:
            deep_state.append(current_state)
    # Reset flag and start iterating until episode ends
    start = time.process_time()
    done = False
    
    while not done:

        global epsilon, EPSILON_DECAY, MIN_EPSILON

        # This part stays mostly the same, the change is to query a model for Q values
        randomChoice = "False"
        if np.random.random() > epsilon:
            # Get action from Q table
            if stateDepth > 1 and len(deep_state) < stateDepth:
                action = np.random.randint(0, ACTION_SPACE_SIZE)
                randomChoice = "True - Depth"
            else:
                action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, ACTION_SPACE_SIZE)
            randomChoice = "True - Epsilon"


        # jd^: start here v
        run_into_self = game.control(action)
        dead, cause = game.update_snake()
        field, reward = game.eat()
        field = game.update_field()
        cpu, ram, step_time, gpu_load, gpu_mem = 0, 0, 0, 0, 0

        new_state = np.array(field) # jd^:

        if useRGBinput:
            new_state = np.array(render.getScreenshot(field))

        if stateDepth > 1 and step_count > 1:
            deep_state.append(new_state)
            if len(deep_state) > stateDepth:
                deep_state.pop(0)
                state_ready = True
            new_state = deep_state

        if not episode % RENDER_EVERY:
            if renderText:
                render.textRender(field, sleepText)
            if renderVisual:
                render.visualRender(field, sleepVisual)

        del field
        
        done = dead
        if dead:
            reward = reward_wall

        elif run_into_self:
            reward = reward_into_self

        if reward == reward_fruit:
            fruit_counter += 1

        if not dead and not run_into_self and reward != reward_fruit:
            reward = reward_step
        
        if reward_distance:
            if step_count == 1:
                pass
            else:
                # reward += math.pow((game.max_distance - game.closest_distance) / game.max_distance, reward_distance_exponent) # NOTE OLD Function
                reward += (prev_dist - game.closest_distance) / reward_distance_exponent
            prev_dist = game.closest_distance

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        if stateDepth == 1 or state_ready:
            agent.update_replay_memory((current_state, action, reward, new_state, done), reward)
            # if not step_count % 4: # TODO make pretty
            agent.train(done, step_count)

        current_state = new_state
        del new_state
        
        step_count += 1
        if(step_count >= MAX_STEPS):
            done = True
        
        if trackCPU_RAM:
            cpu = process.cpu_percent() # if bigger than 100 - multiple threads on multiple cores
            ram = process.memory_percent()
    
        if trackGPU and agent.has_gpu:    
            gpu = GPUtil.getGPUs()
            gpu = gpu[agent.gpu_id]
            gpu_load = gpu.load*100
            gpu_mem = gpu.memoryUtil*100
        step_time = time.process_time() - start
        if(LOG_EVERY_STEP):
            log.log(episode, step_count-1, reward, episode_reward, action, (game.direction), (game.head), (game.closest_fruit), dead, epsilon, run_into_self, 
                    cause, fruit_counter, game.closest_distance, cpu, ram, step_time, gpu_load, gpu_mem, agent.gpu_id, [len(agent.replay_memory), use_good_mem, len(agent.replay_memory_good)], len(deep_state), randomChoice, mode)

        
    # Append episode reward to a list and log stats (every given number of episodes)
    plot.appendLists(episode_reward, epsilon, step_count, fruit_counter, cpu, ram, step_time, gpu_load, gpu_mem)
    if not episode % AGGREGATE_STATS_EVERY:
        plot.averageLists()
        plot.saveFig()
        plot.saveData()
        plot.saveModel(agent.target_model)

    log.log(episode, step_count-1, reward, episode_reward, action, (game.direction), (game.head), (game.closest_fruit), dead, epsilon, run_into_self, 
            cause, fruit_counter, game.closest_distance, cpu, ram, step_time, gpu_load, gpu_mem, agent.gpu_id, [len(agent.replay_memory), use_good_mem, len(agent.replay_memory_good)], len(deep_state), randomChoice, mode)
           
    # Decay epsilon
    if epsilon > MIN_EPSILON and episode > EPISODES_BEFORE_DECAY:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if useRGBinput:
        render.quitPygame()
    elif not episode % RENDER_EVERY:
        if renderVisual:
            render.quitPygame()

    del step_count, reward, episode_reward, action, dead, run_into_self, cause, current_state, ram, cpu, start, step_time, deep_state

# Iterate over episodes
for episode in range(1, EPISODES + 1):
    main(episode)
    tf.keras.backend.clear_session()
    gc.collect()

'''
 TODO:
    1. make more things parameters: Wall Death vs teleport certain reward value or scenarios?
    1. add reward function as in paper
    1. read entire paper, check differences
    .
    .
    .
    99. add more features such as:
        - wall blocks in field (block that spawn random on field like food (but only during init) , but different color that kill the snake
        - other fruit types (for more points, food ranking) - make dynamic if possible
        - diagonal walking
        - multiple snakes
  '''
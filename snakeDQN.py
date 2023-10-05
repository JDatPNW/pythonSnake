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

# class imports
import DQNAgent
import snakeGame
import archiver
import logger

# These are the parameters and settings used in the snake game and the DQN agent. Here is a brief
# explanation of each parameter:
ACTION_SPACE_SIZE = 5 # Number possible actions
WIDTH = 12 # Width of playable field
HEIGHT = 12 # Height of playable field
START_LENGTH = 1 # Starting Length for snake
NUM_FRUIT = (WIDTH+HEIGHT)/2 # NUMBER OF APPLES SPAWNED
CAN_PORT = True # Can the snake come back from the opposite site when hitting the wall?
EPISODES = 50_000 # Number of episodes
DISCOUNT = 0.99 # Discount factor / alpha
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
AGGREGATE_STATS_EVERY = 100  # episodes used for averaging for plotting
LOG_EVERY_STEP = True # Log into console every step?
TF_VERBOSE = True
EXPERIMENT_NAME = "Snake-post-changes"

# The code is creating instances of three different classes: `DQNAgent`, `snakeGame`,, `Archiver` and 'Logger'.
agent = DQNAgent.DQNAgent(REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, UPDATE_TARGET_EVERY, 
                          DISCOUNT, WIDTH, HEIGHT, ACTION_SPACE_SIZE, TF_VERBOSE)
game = snakeGame.snakeGame(WIDTH, HEIGHT, START_LENGTH, NUM_FRUIT, CAN_PORT)
plot = archiver.Archiver(AGGREGATE_STATS_EVERY, EXPERIMENT_NAME, EPISODES)
log = logger.Logger()

epsilon = 1 # Start Value for Epsilon
EPSILON_DECAY = 0.9995 # Rate at which Epsilon decays
MIN_EPSILON = 0.1 # Value where that decay stops

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
    current_state = np.array(list(game.initGame()))

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        global epsilon, EPSILON_DECAY, MIN_EPSILON

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, ACTION_SPACE_SIZE)

        # jd^: start here v
        run_into_self = game.control(action)
        dead, cause = game.update_snake()
        field, reward = game.eat()
        field = game.update_field()

        new_state = np.array(field) # jd^:
        del field
        
        done = dead
        if dead:
            reward = -50

        elif run_into_self:
            reward = -5

        if reward == 25:
            fruit_counter += 1

        if not dead and not run_into_self and reward != 25:
            reward = -0.1

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step_count)

        current_state = new_state
        del new_state
        
        step_count += 1
        if(step_count >= 150):
            done = True
        
        
        if(LOG_EVERY_STEP):
            log.log(episode, step_count, reward, episode_reward, action, (game.direction), (game.SNAKE[0]), dead, epsilon, run_into_self, cause, fruit_counter)

    # Append episode reward to a list and log stats (every given number of episodes)
    plot.appendLists(episode_reward, epsilon, step_count, fruit_counter)
    if not episode % AGGREGATE_STATS_EVERY:
        plot.averageLists()
        plot.saveFig()
    log.log(episode, step_count, reward, episode_reward, action, (game.direction), (game.SNAKE[0]), dead, epsilon, run_into_self, cause, fruit_counter)
           
    # Decay epsilon
    if epsilon > MIN_EPSILON and episode > 31:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    del step_count, reward, episode_reward, action, dead, run_into_self, cause, current_state

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    main(episode)
    tf.keras.backend.clear_session()
    gc.collect()
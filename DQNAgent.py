''' 
██████╗  ██████╗ ███╗   ██╗
██╔══██╗██╔═══██╗████╗  ██║
██║  ██║██║   ██║██╔██╗ ██║
██║  ██║██║▄▄ ██║██║╚██╗██║
██████╔╝╚██████╔╝██║ ╚████║
╚═════╝  ╚══▀▀═╝ ╚═╝  ╚═══╝
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, InputLayer
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import random
import os


# The DQNAgent class is a reinforcement learning agent that uses a Deep Q-Network (DQN) 
class DQNAgent:
    
    def __init__(self, max_mem, min_mem, mini_batch, update_target, discount, width, height, actions, verbose, input_dims, 
                 useRGBinput, good_mem_size_muliplier, good_mem_min_multiplier, good_mem_split, good_mem_threshold, use_good_mem):    
        """
        The above code is the initialization function for a DQNAgent class in Python, which sets up various
        parameters and creates the main and target neural network models.
        
        :param max_mem: The maximum size of the replay memory, which is the buffer that stores the agent's
        experiences for training
        :param min_mem: The `min_mem` parameter represents the minimum number of steps in a memory to start
        training. It determines the minimum number of experiences (state, action, reward, next state) that
        need to be stored in the replay memory before the training process can begin
        :param mini_batch: The `mini_batch` parameter determines the number of steps (samples) to use for
        training in each iteration. It specifies how many steps should be sampled from the replay memory to
        create a mini-batch for training the neural network
        :param update_target: The `update_target` parameter determines how often the target network is
        updated with the weights of the main network. It specifies the number of steps (or episodes) after
        which the target network is updated. For example, if `update_target` is set to 1000, the target
        network will be
        :param discount: The discount factor, also known as the alpha value, determines the importance of
        future rewards in the reinforcement learning algorithm. It is a value between 0 and 1, where 0 means
        the agent only considers immediate rewards and 1 means the agent considers all future rewards
        equally
        :param width: The width parameter represents the width of the playable field in the game or
        environment. It determines the number of columns or cells in the game grid
        :param height: The `height` parameter represents the height of the playable field in the game or
        environment. It is used to define the shape of the input for the convolutional neural network (CNN)
        in the `create_model` method
        :param actions: The `actions` parameter represents the number of possible actions that the agent can
        take in the environment. It is used to determine the size of the output layer of the neural network
        model. Each action corresponds to a different output neuron, and the agent will choose the action
        with the highest output value
        :param verbose: The `verbose` parameter is a boolean value that determines whether or not TensorFlow
        will print the learning status in the console. If `verbose` is set to `True`, TensorFlow will print
        the learning status. If `verbose` is set to `False`, TensorFlow will not print the learning status
        """

        # The code `if not os.path.isdir('models'): os.makedirs('models')` checks if the directory named
        # "models" exists in the current working directory. If the directory does not exist, it creates a new
        # directory named "models". This is done to ensure that the directory exists before saving any models
        # or files related to the models.
        if not os.path.isdir('models'):
            os.makedirs('models')
        
        # For more repetitive results
        # random.seed(31)
        # np.random.seed(31)
        # tf.random.set_seed(31)

        # These variables are used to configure the parameters of the DQNAgent class. Here is a brief
        # explanation of each variable:
        self.DISCOUNT = discount # Discount factor / alpha
        self.REPLAY_MEMORY_SIZE = max_mem  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = min_mem  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = mini_batch  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = update_target  # Terminal states (end of episodes)
        self.WIDTH = width # Width of playable field
        self.HEIGHT = height # Width of playable field
        self.useRGB = useRGBinput
        self.rgb_shape = input_dims
        self.ACTION_SPACE = actions # number of possible actions
        if self.useRGB:
            self.SHAPE = self.rgb_shape
        else:
            self.SHAPE = (self.WIDTH+2, self.HEIGHT+2, 1) # SHAPE for CNN input. +2s are for the overflow/edge  TODO: if remove overflow are, remove +s TODO: if add images, make the depth variable 
        self.VERBOSE = verbose # will tf print learning status in console?
        # Main model
        self.model = self.create_model()
        
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.good_mem_size_muliplier = good_mem_size_muliplier
        self.good_mem_min_multiplier = good_mem_min_multiplier
        self.good_mem_split = good_mem_split
        self.good_mem_threshold = good_mem_threshold
        self.use_good_mem = use_good_mem
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.replay_memory_good = deque(maxlen=int(self.REPLAY_MEMORY_SIZE*good_mem_size_muliplier))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
        # NOTE!! this was only tested with 1 GPU, might have to be modified if using more than one or if more than 1 are available
        self.gpu_info = tf.config.list_physical_devices('GPU')
        if not self.gpu_info:
            self.has_gpu = False
            self.gpu_id = "NO_GPU"
        else:
            self.has_gpu = True
            self.gpu_id = int(self.gpu_info[0][0][-1])

    def create_model(self):
        """
        The `create_model` function creates a convolutional neural network model for image classification
        with two convolutional layers, max pooling, dropout, and dense layers.
        :return: The function `create_model` returns a compiled Keras model.
        """
        model = Sequential()
        model.add(InputLayer(input_shape=self.SHAPE))
        model.add(Conv2D(16, (3, 3), activation='relu'))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Dropout(0.2))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(Dropout(0.2))
        # model.add(Conv2D(128, (3, 3), activation='relu',strides=1, padding="SAME"))
        # model.add(Dropout(0.2))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.ACTION_SPACE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        print("Input Shape: ", self.SHAPE)
        model.summary()
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition, reward):
        """
        The function `update_replay_memory` appends a transition to the replay memory.
        
        :param transition: The parameter "transition" represents a single transition in a reinforcement
        learning setting. It typically consists of four components: the current state, the action taken, the
        reward received, and the next state. These components are often represented as a tuple or a
        dictionary
        """
        if reward > self.good_mem_threshold and self.use_good_mem: # TODO make pretty
            self.replay_memory_good.append(transition)
        else:
            self.replay_memory.append(transition)
            
    # Trains main network every step during episode
    def train(self, terminal_state, step):
        """
        The `train` function is used to train a neural network model using a replay memory and a target
        network.
        
        :param terminal_state: The terminal_state parameter is a boolean value that indicates whether the
        current state is a terminal state or not. In reinforcement learning, a terminal state is a state in
        which the episode ends and no further actions can be taken
        :param step: The `step` parameter represents the current step or iteration in the training process.
        It is used to keep track of the progress and determine when to update the target network
        :return: The function does not return anything.
        """

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE: # TODO: make pretty
            return
        if self.use_good_mem:
            if len(self.replay_memory_good) < int(self.MIN_REPLAY_MEMORY_SIZE*self.good_mem_min_multiplier):
                return

        # Get a minibatch of random samples from memory replay table
        if self.use_good_mem:
            minibatch = random.sample(self.replay_memory, int(self.MINIBATCH_SIZE*(1-self.good_mem_split))) # TODO: make pretty
            minibatch += random.sample(self.replay_memory_good, int(self.MINIBATCH_SIZE*self.good_mem_split)) # TODO: make pretty
        else:
            minibatch = random.sample(self.replay_memory, int(self.MINIBATCH_SIZE)) # TODO: make pretty
        

        # Get current states from minibatch, then query NN model for Q values        
        current_states = np.array([transition[0] for transition in minibatch])/255
        
        if not self.useRGB:
            current_states = np.expand_dims(current_states, -1) 
        
        current_qs_list = self.model.predict(current_states, verbose=self.VERBOSE)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states, verbose=self.VERBOSE)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=self.VERBOSE, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        del minibatch, current_qs, current_state, current_qs_list, new_current_states, future_qs_list, X, y
        
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        """
        The function takes a state as input, converts it to a numpy array, reshapes it, and then uses a
        model to predict the Q-values for that state.
        
        :param state: The `state` parameter is a list or array representing the current state of the
        environment or game
        :return: the predicted output of the model for the given state.
        """
        state = np.array(state).reshape(-1, *self.SHAPE)/255
        return self.model.predict(state, verbose=self.VERBOSE)[0]
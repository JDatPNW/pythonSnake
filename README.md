# pythonSnake

                    /^\/^\           ███████╗███╗   ██╗ █████╗ ██╗  ██╗███████╗    ██████╗  ██████╗ ███╗   ██╗        . ~ ------- ~ .
                  __|_| O|  \        ██╔════╝████╗  ██║██╔══██╗██║ ██╔╝██╔════╝    ██╔══██╗██╔═══██╗████╗  ██║      .'  .~ _____ ~-. `.
           \/    /~    \_/    \      ███████╗██╔██╗ ██║███████║█████╔╝ █████╗█████╗██║  ██║██║   ██║██╔██╗ ██║    /   /             `.`.
            \____|__________/   _    ╚════██║██║╚██╗██║██╔══██║██╔═██╗ ██╔══╝╚════╝██║  ██║██║▄▄ ██║██║╚██╗██║   /   /                `'
                    \_______         ███████║██║ ╚████║██║  ██║██║  ██╗███████╗    ██████╔╝╚██████╔╝██║ ╚████║ /    |
                            \ __     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝  ╚══▀▀═╝ ╚═╝  ╚═══╝

# Snake Game with Deep Q-Network (DQN) Agent

## Introduction

This project is an implementation of the classic Snake game using a Deep Q-Network (DQN) reinforcement learning agent. The DQN agent learns to play the game by interacting with the environment, making decisions, and updating its Q-values based on rewards.

## Table of Contents
- [Introduction](#introduction)
- [How to Play](#how-to-play)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training the DQN Agent](#training-the-dqn-agent)
- [Visualization and Logging](#visualization-and-logging)
- [Contributing](#contributing)
- [License](#license)

## How to Play

The Snake game can be played as follows:

- The snake starts with a length of 5 segments.
- The goal is to collect fruits (green cells) to increase the snake's length.
- Avoid running into walls and the snake's own body.
- The game ends if the snake collides with a wall or itself.

## Project Structure

The project consists of the following key components:

- `DQNAgent.py`: The DQN agent implementation for training and making decisions.
- `snakeGame.py`: The Snake game simulator.
- `archiver.py`: A class for archiving game statistics and generating plots.
- `logger.py`: A logger class for recording game and agent information.
- `snakeDQN.py`: The main script for training the DQN agent and running game episodes.
- `plots/`: A directory for storing generated plots.
- `LICENSE`: The project's open-source license file.

## Getting Started

1. Clone this repository to your local machine:

   ```shell
   git clone https://github.com/yourusername/snake-dqn.git
   cd snake-dqn
   ```
2. Ensure you have the required Python libraries installed. You can install them using pip:

  ```shell
    pip install numpy tensorflow tqdm colorama matplotlib psutil
  ```
3. Start the Snake game with the DQN agent by running the following command:
  ```shell
    python snakeDQN.py
  ```
4. Follow the console log to observe the training process and agent's performance.

## Training the DQN Agent
The DQN agent uses reinforcement learning to improve its performance over time. Key training parameters and settings can be customized in snakeDQN.py. These include:

- The number of training episodes (EPISODES).
- The replay memory size (REPLAY_MEMORY_SIZE).
- The target network update frequency (UPDATE_TARGET_EVERY).
- Epsilon-greedy exploration settings (epsilon, EPSILON_DECAY, MIN_EPSILON).
- Other training and game parameters.
- Feel free to modify these parameters to experiment with training dynamics and agent performance.

## Visualization and Logging
The project includes utilities for visualizing and logging the game and agent performance:

- The archiver.py class generates plots to visualize statistics, including average rewards, steps, and fruit counts.
- The logger.py class logs game and agent information, making it easy to track the agent's behavior during training.

Plots are saved in the plots/ directory, and log information is displayed in the console.

## Contributing
Contributions to this project are welcome! You can contribute by reporting issues, proposing enhancements, or submitting pull requests. Please review the project's contribution guidelines for details.

## License
This project is open-source and available under the MIT License. You are free to use, modify, and distribute this code as allowed by the license.

Have fun playing Snake and training your DQN agent!





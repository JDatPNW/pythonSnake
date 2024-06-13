import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Define color palettes for different iterations
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

directory  = "./special_cases/"
data_dict = {
    "filename" : [],
    "data" : []
}
dirlsit = os.listdir(directory)
dirlsit = sorted(dirlsit)

for file in dirlsit:
    if file.endswith(".npy"):
        print(file)
        data_dict["filename"].append(file)
        data = np.load(directory+file, allow_pickle=True).item()
        data_dict["data"].append(data)


fig, ax1 = plt.subplots(figsize=(20, 8))
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

for i, _ in enumerate(data_dict["filename"]):
    print(i)
    parts = data_dict["filename"][i].split('_')  # Split the filename at underscores
    print(parts)
    label = ' '.join([parts[0].capitalize(), parts[1], parts[2].capitalize(), parts[3], parts[4].capitalize(), parts[5].capitalize(), parts[6].capitalize()])  # Join the first two parts with a space and capitalize the first part
    if(parts[-2]) == "BW":
        label += " BW"
    average_reward_list = data_dict["data"][i]["average_reward_list"]
    min_reward_list = data_dict["data"][i]["min_reward_list"]
    max_reward_list = data_dict["data"][i]["max_reward_list"]
    average_step_list = data_dict["data"][i]["average_step_list"]
    average_fruit_list = data_dict["data"][i]["average_fruit_list"]
    average_epsilon_list = data_dict["data"][i]["average_epsilon_list"]
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward/Steps')
    
    # Selecting colors from the predefined palettes based on loop index
    # color_palette = plt.get_cmap(color_palettes[i % len(color_palettes)])

    ax1.plot(average_reward_list, label="Average Reward " + label, color=color_palette[i])
    # ax1.plot([100 * x for  x in average_epsilon_list], label = "epsilon percentage " + label, color=color_palette(0.7))
    # ax1.plot(min_reward_list, label = "min. Reward " + label, color=color_palette(0.5))
    # ax1.plot(max_reward_list, label = "max. Reward " + label, color=color_palette(0.1))
    # ax1.plot(average_step_list, label = "Average Steps " + label, color=color_palette(0.9))

    # ax2.set_ylabel('Fruits Eaten')  # we already handled the x-label with ax1
    # ax2.plot(average_fruit_list, label = "Average Fruits eaten " + label, color=color_palette(0.6))
    # ax2.tick_params(axis='y')


# Add overall legend for all plots
max_value = max([len(data_dict["data"][i]["average_reward_list"]) for i in range(len(data_dict["filename"]))])

ax1.set_ylim([-100, 400])
# ax2.set_ylim([0, 12])

ax1.set_xlim([0, max_value])
ax1.set_xticks(np.arange(0, max_value+1, 10))  # Adjust the step size to display all 10 steps
ax1.set_xticklabels(np.arange(0, max_value+1, 10) * 100)  # Multiply the tick labels by 100

ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
plt.savefig("./special_cases/" + "test" + ".png", dpi=300)


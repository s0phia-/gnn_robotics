import os
import numpy as np
import matplotlib.pyplot as plt


def read_reward_data(file_path):
    """Read reward data from CSV file"""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]  # iterations, rewards


def collect_data_from_folder(results_folder):
    """Collect data from all CSV files in folder"""
    csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
    data_dict = {}
    for csv_file in csv_files:
        file_path = os.path.join(results_folder, csv_file)
        param_name = os.path.splitext(csv_file)[0]
        if param_name.startswith('rewards_'):
            param_name = param_name[8:]
        iterations, rewards = read_reward_data(file_path)
        data_dict[param_name] = (iterations, rewards)
    return data_dict


def plot_reward_data(data_dict, save_path=None):
    """Plot collected reward data"""
    plt.figure(figsize=(10, 6))
    for param_name, (iterations, rewards) in data_dict.items():
        plt.plot(iterations, rewards, label=param_name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_rewards_from_folder(results_folder, save_path=None):
    """Main function to read and plot reward data"""
    data_dict = collect_data_from_folder(results_folder)
    plot_reward_data(data_dict, save_path)

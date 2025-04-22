import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re


def read_reward_data(file_path):
    """Read reward data from CSV file"""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]  # iterations, rewards


def extract_params(filename):
    """Extract parameter values from filename, excluding seed"""
    # Extract all param-value pairs
    params = re.findall(r'([a-zA-Z_]+)-([0-9.]+)', filename)
    # Filter out the seed parameter
    return tuple((p, v) for p, v in params if p != 'seed')


def collect_data_with_seeds(results_folder):
    """Group data by parameters (excluding seed)"""
    csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
    grouped_data = defaultdict(list)
    for csv_file in csv_files:
        file_path = os.path.join(results_folder, csv_file)
        param_tuple = extract_params(csv_file)
        iterations, rewards = read_reward_data(file_path)
        grouped_data[param_tuple].append((iterations, rewards))
    return grouped_data


def compute_statistics(grouped_data):
    """Compute average and std dev for each parameter group"""
    avg_data = {}
    for param_tuple, runs in grouped_data.items():
        # Create param name string for plotting
        param_name = "_".join([f"{p}-{v}" for p, v in param_tuple])

        # Find min length to align data
        min_length = min(len(run[1]) for run in runs)
        aligned_rewards = np.array([run[1][:min_length] for run in runs])
        iterations = runs[0][0][:min_length]

        mean_rewards = np.mean(aligned_rewards, axis=0)
        std_rewards = np.std(aligned_rewards, axis=0)
        avg_data[param_name] = (iterations, mean_rewards, std_rewards)
    return avg_data


def plot_averaged_data(avg_data, save_path=None):
    """Plot averaged reward data with error bands"""
    plt.figure(figsize=(10, 6))
    for param_name, (iterations, mean_rewards, std_rewards) in avg_data.items():
        plt.plot(iterations, mean_rewards, label=param_name, linewidth=2)
        plt.fill_between(iterations,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards Over Seeds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_rewards_with_seeds(results_folder, save_path=None):
    """Main function to read, average, and plot reward data"""
    grouped_data = collect_data_with_seeds(results_folder)
    avg_data = compute_statistics(grouped_data)
    plot_averaged_data(avg_data, save_path)

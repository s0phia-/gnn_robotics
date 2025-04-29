import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict


def process_folder(folder_path):
    results = defaultdict(list)
    for filename in os.listdir(folder_path):
        if not ',' in filename or filename.startswith('.'):
            continue
        param_value_pairs = filename.split(',')
        params_dict = {}
        for pair in param_value_pairs:
            parts = pair.split('-', 1)
            if len(parts) == 2:
                param_name, param_value = parts
                if param_name != 'seed':
                    params_dict[param_name] = param_value
        param_key = tuple(sorted(params_dict.items()))
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            rewards = [float(row[1]) for row in reader]
        results[param_key].append(rewards)
    return {', '.join(f"{name}: {value}" for name, value in key): values
            for key, values in results.items()}


def average_results(results_dict):
    averaged_results = {}
    for param_str, list_of_lists in results_dict.items():
        averaged_results[param_str] = np.mean(np.array(list_of_lists), axis=0)
    return averaged_results


def plot_averaged_data(avg_data, save_path=None, smoothed=False, show_std=False):
    plt.figure(figsize=(12, 8))
    param_names = sorted(avg_data.keys())
    for param_name in param_names:
        mean_rewards = avg_data[param_name]
        iterations = range(len(mean_rewards))
        if smoothed:
            smoothed_rewards = np.convolve(mean_rewards, np.ones(5)/5, mode='valid')
            smoothed_iterations = iterations[2:-2]
            plt.plot(smoothed_iterations, smoothed_rewards, label=param_name, linewidth=2)
        else:
            plt.plot(iterations, mean_rewards, label=param_name, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Training Rewards Averaged Over Seeds', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rewards_with_seeds(results_folder, save_path=None):
    grouped_data = process_folder(results_folder)
    avg_data = average_results(grouped_data)
    # avg_data.pop("decoder_and_message_hidden_dim-64_hidden_node_dim-32_propagation_steps-4", None)
    # avg_data.pop("decoder_and_message_hidden_dim-64_hidden_node_dim-64_propagation_steps-4", None)
    # avg_data.pop("decoder_and_message_hidden_dim-128_hidden_node_dim-32_propagation_steps-4", None)
    # avg_data.pop("decoder_and_message_hidden_dim-128_hidden_node_dim-64_propagation_steps-4", None)
    plot_averaged_data(avg_data, save_path)
    return avg_data

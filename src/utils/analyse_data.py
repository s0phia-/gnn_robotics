import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import json


def read_reward_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


def get_varying_params(hparams_list):
    varying_params = {}
    all_params = set()
    for config in hparams_list:
        all_params.update(config.keys())
    for param in all_params:
        if param in ['seed', 'run_id', 'run_dir']:
            continue
        values = set()
        for config in hparams_list:
            if param in config:
                if isinstance(config[param], (list, dict)):
                    values.add(json.dumps(config[param]))
                else:
                    values.add(config[param])
        if len(values) > 1:
            varying_params[param] = list(values)
    return varying_params


def extract_params_from_filename(filename, varying_params):
    if filename.endswith('.csv'):
        filename = filename[:-4]
    params = {}
    seed = None
    for param in varying_params:
        pattern = f"{param}-([^_]+)"
        match = re.search(pattern, filename)
        if match:
            value = match.group(1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            params[param] = value
    seed_match = re.search(r"seed-([^_]+)", filename)
    if seed_match:
        seed = seed_match.group(1)
    return params, seed


def collect_data_with_seeds(results_folder, hparams_list):
    varying_params = get_varying_params(hparams_list)
    csv_files = [f for f in os.listdir(results_folder) if f.endswith('.csv')]
    grouped_data = defaultdict(list)
    for csv_file in csv_files:
        file_path = os.path.join(results_folder, csv_file)
        params, seed = extract_params_from_filename(csv_file, varying_params)
        if not params:
            continue
        if seed is None:
            seed = "unknown"
        param_key = tuple(sorted((k, v) for k, v in params.items()))
        iterations, rewards = read_reward_data(file_path)
        if len(rewards) == 0:
            continue
        grouped_data[param_key].append((iterations, rewards, seed))
    return grouped_data


def compute_statistics(grouped_data):
    avg_data = {}
    for param_tuple, runs in grouped_data.items():
        param_name = "_".join([f"{p}-{v}" for p, v in param_tuple])
        if not runs:
            continue
        min_length = min(len(run[1]) for run in runs)
        if min_length == 0:
            continue
        aligned_rewards = np.array([run[1][:min_length] for run in runs])
        iterations = runs[0][0][:min_length]
        mean_rewards = np.mean(aligned_rewards, axis=0)
        std_rewards = np.std(aligned_rewards, axis=0)
        avg_data[param_name] = (iterations, mean_rewards, std_rewards)
    return avg_data


def plot_averaged_data(avg_data, save_path=None, smoothed=False, show_std=False):
    plt.figure(figsize=(12, 8))
    if avg_data:
        param_names = sorted(avg_data.keys())
        for param_name in param_names:
            iterations, mean_rewards, std_rewards = avg_data[param_name]
            if smoothed:
                smoothed_rewards = np.convolve(mean_rewards, np.ones(5)/5, mode='valid')
                smoothed_iterations = iterations[2:-2]
                plt.plot(smoothed_iterations, smoothed_rewards, label=param_name, linewidth=2)
            else:
                plt.plot(iterations, mean_rewards, label=param_name, linewidth=2)
            if show_std:
                plt.fill_between(iterations,
                                 mean_rewards - std_rewards,
                                 mean_rewards + std_rewards,
                                 alpha=0.2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Training Rewards Averaged Over Seeds', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rewards_with_seeds(results_folder, hparams_list, save_path=None):
    grouped_data = collect_data_with_seeds(results_folder, hparams_list)
    avg_data = compute_statistics(grouped_data)
    # avg_data.pop("decoder_and_message_hidden_dim-64_hidden_node_dim-32_propagation_steps-4", None)
    # avg_data.pop("decoder_and_message_hidden_dim-64_hidden_node_dim-64_propagation_steps-4", None)
    # avg_data.pop("decoder_and_message_hidden_dim-128_hidden_node_dim-32_propagation_steps-4", None)
    # avg_data.pop("decoder_and_message_hidden_dim-128_hidden_node_dim-64_propagation_steps-4", None)
    plot_averaged_data(avg_data, save_path)
    return avg_data

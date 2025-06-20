import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict

import re


def process_folder(folder_path):
    results = defaultdict(list)
    for filename in os.listdir(folder_path):
        if filename.startswith('.') or not filename.endswith('.csv'):
            continue

        # Remove .csv extension first
        filename_no_ext = filename.replace('.csv', '')

        # Use regex to find all parameter-value pairs
        # Pattern: word characters, underscores, dashes followed by dash and value
        pattern = r'([a-zA-Z_]+)-([^_,]+)'
        matches = re.findall(pattern, filename_no_ext)

        params_dict = {}

        for param_name, param_value in matches:
            if param_name not in ['seed', '_seed']:  # Exclude both 'seed' and '_seed' from grouping
                params_dict[param_name] = param_value

        param_key = tuple(sorted(params_dict.items()))
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                rewards = [float(row[1]) for row in reader if len(row) > 1]  # Added safety check
            results[param_key].append(rewards)
        except Exception as e:
            continue

    return {', '.join(f"{name}: {value}" for name, value in key): values
            for key, values in results.items()}


# def average_results(results_dict):
#     averaged_results = {}
#     for param_str, list_of_lists in results_dict.items():
#         averaged_results[param_str] = np.mean(np.array(list_of_lists), axis=0)
#     return averaged_results


def average_results(results_dict):
    """Fixed averaging function that handles different length arrays"""
    averaged_results = {}

    for param_str, list_of_arrays in results_dict.items():

        if not list_of_arrays:
            continue

        lengths = [len(arr) for arr in list_of_arrays]

        if len(set(lengths)) > 1:
            min_length = min(lengths)
            trimmed_arrays = [arr[:min_length] for arr in list_of_arrays]
            list_of_arrays = trimmed_arrays

        # Convert to numpy array and average
        try:
            array_2d = np.array(list_of_arrays)  # Shape: (num_seeds, num_timesteps)
            averaged_results[param_str] = np.mean(array_2d, axis=0)  # Average across seeds
        except Exception as e:
            averaged_results[param_str] = np.array(list_of_arrays[0])

    return averaged_results


def plot_averaged_data(avg_data, save_path=None, smoothed=False):
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


def filter_top_n(data_dict, n=1):
    """One-liner to get top N keys"""
    final_avgs = {k: np.mean(v[-100:]) for k, v in data_dict.items()}
    top_keys = sorted(final_avgs.keys(), key=lambda k: final_avgs[k], reverse=True)[:n]
    return {k: data_dict[k] for k in top_keys}


def filter_top_n_by_method(data_dict, n=3):
    """Get top N keys grouped by method"""
    final_avgs = {k: np.mean(v[-100:]) for k, v in data_dict.items()}

    # Group by method
    from collections import defaultdict
    methods = defaultdict(list)
    for k in final_avgs:
        method = k.split('method: ')[1].split(',')[0]
        methods[method].append(k)

    # Get top n from each method
    result = {}
    for method, keys in methods.items():
        top_key = sorted(keys, key=lambda k: final_avgs[k], reverse=True)[0]
        result[method] = data_dict[top_key]

    return result


def plot_rewards_with_seeds(results_folder):
    save_path=results_folder
    grouped_data = process_folder(results_folder)
    plt_data = average_results(grouped_data)
    plt_data = filter_top_n_by_method(plt_data)
    plot_averaged_data(plt_data, save_path)
    return plt_data

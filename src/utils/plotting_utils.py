import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import os
from collections import defaultdict


def process_folder(folder_path):
    results = defaultdict(list)
    for filename in os.listdir(folder_path):
        filename_no_ext = filename.replace('.csv', '')
        if filename_no_ext.startswith(',seed-'):
            seed_match = re.search(r',seed-(\d+)', filename_no_ext)
            if seed_match:
                param_key = tuple()  # Empty tuple groups all seed files together
        else:
            pattern = r'([a-zA-Z_]+)-([^_,]+)'
            matches = re.findall(pattern, filename_no_ext)
            params_dict = {}
            for param_name, param_value in matches:
                if param_name not in ['seed', '_seed']:
                    params_dict[param_name] = param_value
            param_key = tuple(sorted(params_dict.items()))
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rewards = [float(row[1]) for row in reader if len(row) > 1]
            results[param_key].append(rewards)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue
    final_results = {}
    for key, values in results.items():
        if key == tuple():  # Empty tuple for seed-only files
            final_results["seed_only_runs"] = values
        else:
            final_results[', '.join(f"{name}: {value}" for name, value in key)] = values
    return final_results


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
        try:
            array_2d = np.array(list_of_arrays)
            averaged_results[param_str] = np.mean(array_2d, axis=0)
        except:
            averaged_results[param_str] = np.array(list_of_arrays[0])
    return averaged_results


def plot_averaged_data(avg_data, save_path=None, smoothed=False, window_size=None):
    plt.figure(figsize=(12, 8))
    param_names = sorted(avg_data.keys())
    for param_name in param_names:
        mean_rewards = avg_data[param_name]
        iterations = range(len(mean_rewards))
        if smoothed:
            smoothed_rewards = np.convolve(mean_rewards, np.ones(window_size) / window_size, mode='valid')
            smoothed_iterations = range(window_size - 1, len(mean_rewards))
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


def plot_rewards_with_seeds(folder_path):
    save_path = folder_path
    grouped_data = process_folder(folder_path)
    print(grouped_data)
    plt_data = average_results(grouped_data)
    plot_averaged_data(plt_data, save_path, smoothed=True, window_size=10)
    return plt_data


if __name__ == '__main__':
    results_folder = '../../runs/gnn/results'
    plot_rewards_with_seeds(results_folder)

import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
import re
import os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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
    # plt_data = filter_top_n_by_method(plt_data)
    plot_averaged_data(plt_data, save_path, smoothed=True, window_size=10)
    return plt_data


def extract_skrl_tensorboard_data(base_path, metrics=None, use_mean_for_aggregated=True):
    """Extract TensorBoard data from SKRL experiment structure."""
    if metrics is None:
        metrics = ['Reward / Instantaneous reward', 'Reward / Total reward', 'Loss / Policy loss',
                   'Loss / Value loss', 'Policy / Standard deviation', 'Episode / Total timesteps']

    skrl_data = defaultdict(dict)

    # More flexible pattern matching for different folder structures
    patterns = [
        os.path.join(base_path, "ppo", "*", "*", "seed_*", "*", "events.out.tfevents.*"),
        os.path.join(base_path, "ppo", "*", "*", "*", "seed_*", "*", "events.out.tfevents.*"),
        os.path.join(base_path, "**", "events.out.tfevents.*")
    ]

    event_files = []
    for pattern in patterns:
        event_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates
    event_files = list(set(event_files))

    if not event_files:
        return dict(skrl_data)

    for event_file in event_files:
        try:
            path_parts = event_file.split(os.sep)

            # Find ppo index
            if 'ppo' not in path_parts:
                continue

            ppo_idx = path_parts.index('ppo')

            # Extract components more flexibly
            env_name = None
            method_name = None
            seed_number = 0

            # Look for seed folder and work backwards
            seed_idx = None
            for i, part in enumerate(path_parts):
                if part.startswith('seed_'):
                    seed_idx = i
                    seed_number = int(part.split('_')[-1])
                    break

            if seed_idx is None:
                continue

            # Extract env and method based on position relative to ppo and seed
            remaining_parts = path_parts[ppo_idx + 1:seed_idx]

            if len(remaining_parts) >= 1:
                env_name = remaining_parts[0]
            if len(remaining_parts) >= 2:
                method_name = remaining_parts[-1]  # Take the last part before seed as method

            # If we have more parts, create a combined method name
            if len(remaining_parts) > 2:
                method_name = "_".join(remaining_parts[1:])

            # Default values if not found
            if not env_name:
                env_name = "unknown_env"
            if not method_name:
                method_name = "unknown_method"

            config_key = f"{method_name}_seed_{seed_number:03d}"

            event_acc = EventAccumulator(os.path.dirname(event_file))
            event_acc.Reload()
            available_tags = event_acc.Tags()['scalars']

            if config_key not in skrl_data[env_name]:
                skrl_data[env_name][config_key] = {'seed': seed_number, 'method': method_name, 'env': env_name}

            for metric in metrics:
                data_points = []
                if metric in available_tags:
                    data_points = [(event.step, event.value) for event in event_acc.Scalars(metric)]
                else:
                    matches = [tag for tag in available_tags if metric in tag]
                    if matches:
                        chosen = next((m for m in matches if '(mean)' in m), matches[0]) if use_mean_for_aggregated else \
                        matches[0]
                        data_points = [(event.step, event.value) for event in event_acc.Scalars(chosen)]
                skrl_data[env_name][config_key][metric] = data_points
        except:
            continue

    return dict(skrl_data)


def aggregate_seeds(skrl_data, aggregation='mean'):
    """Aggregate data across seeds for the same method/environment combination."""
    aggregated_data = defaultdict(dict)

    for env_name, env_data in skrl_data.items():
        method_groups = defaultdict(list)
        for config_key, config_data in env_data.items():
            method_groups[config_data['method']].append(config_data)

        for method, configs in method_groups.items():
            if aggregation == 'all':
                for config in configs:
                    aggregated_data[env_name][f"{method}_seed_{config['seed']:03d}"] = config
            else:
                aggregated_config = {'method': method, 'env': env_name, 'seeds': [c['seed'] for c in configs]}
                all_metrics = set().union(*[set(c.keys()) for c in configs]) - {'seed', 'method', 'env'}

                for metric in all_metrics:
                    metric_data = [c[metric] for c in configs if metric in c and c[metric]]
                    if metric_data:
                        all_steps = sorted(set().union(*[{step for step, _ in data} for data in metric_data]))
                        interpolated = []
                        for data in metric_data:
                            if data:
                                steps, values = zip(*data)
                                interpolated.append(np.interp(all_steps, steps, values))
                        if interpolated:
                            agg_values = np.mean(interpolated, axis=0) if aggregation == 'mean' else np.median(
                                interpolated, axis=0)
                            aggregated_config[metric] = list(zip(all_steps, agg_values))
                        else:
                            aggregated_config[metric] = []
                    else:
                        aggregated_config[metric] = []
                aggregated_data[env_name][method] = aggregated_config

    return dict(aggregated_data)


def plot_skrl_data(skrl_data, metric='Reward / Total reward', smoothing_window=100, figsize=(15, 10)):
    """Plot SKRL tensorboard data"""
    plt.figure(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx, plotted_any = 0, False

    for exp_name, exp_data in skrl_data.items():
        for config_name, config_data in exp_data.items():
            if metric in config_data and config_data[metric]:
                steps, values = zip(*sorted(config_data[metric], key=lambda x: x[0]))

                if smoothing_window > 1 and len(values) > smoothing_window:
                    smoothed = np.convolve(values, np.ones(smoothing_window) / smoothing_window, mode='valid')
                    plt.plot(steps[smoothing_window - 1:], smoothed, label=f"{exp_name}_{config_name}",
                             color=colors[color_idx % len(colors)], linewidth=2)
                else:
                    plt.plot(steps, values, label=f"{exp_name}_{config_name}",
                             color=colors[color_idx % len(colors)], linewidth=2)

                color_idx += 1
                plotted_any = True

    if not plotted_any:
        plt.text(0.5, 0.5, f"No data found for metric:\n'{metric}'",
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    else:
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'SKRL Training Data: {metric}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)

    plt.tight_layout()
    plt.show()


def list_available_metrics(skrl_data):
    """Print all available metrics in the data"""
    for env_name, env_data in skrl_data.items():
        for config_name, config_data in env_data.items():
            for metric, data in config_data.items():
                if metric not in ['seed', 'method', 'env']:
                    status = "✅" if data else "❌"
                    print(f"{status} '{metric}': {len(data) if data else 0} points")
            break
        break


def analyze_skrl_experiment(base_path, aggregate_seeds_flag=True, metrics_to_plot=None):
    """Complete analysis pipeline for SKRL experiments"""
    skrl_data = extract_skrl_tensorboard_data(base_path)
    if not skrl_data:
        return None, None

    aggregated_data = aggregate_seeds(skrl_data, aggregation='mean') if aggregate_seeds_flag else skrl_data

    if metrics_to_plot is None:
        metrics_to_plot = ['Reward / Total reward']

    for metric in metrics_to_plot:
        plot_skrl_data(aggregated_data, metric=metric)

    return skrl_data, aggregated_data

if __name__ == "__main__":
    runs_path = '/Users/sj2073/Documents/Projects/gnn_robotics/runs/exp_20250725_181754'
    # Complete analysis
    raw_data, aggregated_data = analyze_skrl_experiment(runs_path)

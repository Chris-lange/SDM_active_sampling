import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_loc', type=str, default="")
    parser.add_argument('--load_dir', type=str, default="")
    parser.add_argument('--title', type=str, default="")
    args = vars(parser.parse_args())

    print("This is a small script to generate per iteration figures from the paper. You may have to adjust "
          "various matplotlib and other settings in for perfect results.")

    # Create separate dictionary for each experiment's results
    exp_dict = defaultdict(list)

    for filename in os.listdir(args['load_dir']):
        # Ignore non .npz files
        if not filename.endswith('.npz'):
            continue
        if "IUCN" in filename:
            # Get experiment name and run number from filename
            experiment_name, run_number = filename.rsplit('_IUCN_', 1)
            run_number = int(run_number.rsplit('.', 1)[0])
        elif "SNT" in filename:
            experiment_name, run_number = filename.rsplit('_SNT_', 1)
            run_number = int(run_number.rsplit('.', 1)[0])
        else:
            print("Unexpected file naming format")
            print("add a new elif to the statement with your naming convention to continue")
            raise NotImplementedError


        file_loc = os.path.join(args['load_dir'], filename)
        file = np.load(file_loc, allow_pickle=True)
        results = file['results'].item()

        map_per_it = [results[i]['mean_average_precision'] for i in range(len(results))]

        # Append to this experiment's results
        exp_dict[experiment_name].append(map_per_it)

    # Compute mean and standard error for each experiment
    mean_dict = {}
    std_err_dict = {}

    # Find the maximum length of runs
    max_length = max(len(run) for runs in exp_dict.values() for run in runs)

    # so plotting is possible for when you haven't finished all your runs yet
    for experiment_name, results in exp_dict.items():
        # Create a 2D list filled with None for missing values
        results_padded = [run + [None] * (max_length - len(run)) for run in results]

        # Transpose the 2D list
        results_transposed = list(map(list, zip(*results_padded)))

        # Calculate mean and standard error for each time step
        mean_dict[experiment_name] = [np.mean([x for x in timestep if x is not None]) for timestep in results_transposed]
        std_err_dict[experiment_name] = [
            np.std([x for x in timestep if x is not None])if len(
                [x for x in timestep if x is not None]) > 1 else 0 for timestep in results_transposed]

    # Set style for seaborn
    sns.set_theme(style="whitegrid")

    # For setting the order of the experiments and corresponding labels
    # Fill this in with the names of your experiments and the order you want them to appear in the legend
    # experiment_order = ['WA_HSS_PLUS', 'WA_HSS', 'WA_UNC', 'WA_RANDOM', 'LR_HSS', 'LR_UNC', 'LR_RANDOM']
    experiment_order = ['LR_RANDOM']

    # To translate from experiment file names to the name you want in the legend
    experiment_labels = {
        'WA_HSS_PLUS': 'WA_HSS+',
        'WA_UNC_PLUS': 'WA_uncertain+',
        'WA_HSS': 'WA_HSS',
        'WA_RANDOM': 'WA random',
        'WA_UNC': 'WA_uncertain',
        'LR_HSS': 'LR_HSS',
        'LR_RANDOM': 'LR_random',
        'LR_UNC': 'LR_uncertain',
    }

    # Create a color palette
    palette = sns.color_palette("bright", len(experiment_order))
    # set serif font
    plt.rcParams['font.family'] = 'serif'
    # Plot the baseline if available
    an_full_1000_map = 0.73581165 # THIS IS CURRENTLY OUTDATED - maybe replace with uncapped as well.
    plt.plot([an_full_1000_map]*100, linestyle='--', color='k',  label='AN_FULL_E2E_OUTDATED')

    # plot the results
    for i, experiment_name in enumerate(experiment_order):
        mean_results = np.array(mean_dict[experiment_name])
        std_err_results = np.array(std_err_dict[experiment_name])
        x = np.array(range(len(mean_results)))

        # fiddle with the below to determine line style for different kinds of experiments
        if "LR" in experiment_name:
            linestyle = '--'
        else:
            linestyle = '-'

        label = experiment_labels[experiment_name]
        plt.plot(x, mean_results, label=label, color=palette[i], linestyle=linestyle)
        plt.fill_between(x, mean_results - std_err_results, mean_results + std_err_results, color=palette[i], alpha=0.2)

    # Set the labels and title
    plt.xlabel('Time')
    plt.ylabel('MAP')
    plt.xlim(0, 50)
    plt.ylim(0.3, 0.9)
    plt.title(f'{args["title"]}', fontsize = 14)

    # Draw the legend and save the plot
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    pdf_save_loc = args["save_loc"] + '.pdf'
    png_save_loc = args["save_loc"] + '.png'
    plt.savefig(pdf_save_loc)
    plt.savefig(png_save_loc)
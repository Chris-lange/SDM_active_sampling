import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import argparse

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="saved_figures")
    parser.add_argument('--save_name', type=str, default="histogram_WA_RANDOM")
    parser.add_argument('--load_loc', type=str, default="../results/all_results/WA_RANDOM_IUCN_3.npz")
    parser.add_argument('--title', type=str, default="per species AP Histogram")
    args = vars(parser.parse_args())

    print("This is a small script to generate histogram figures from the paper. You may have to adjust "
          "various matplotlib and other settings in this script for perfect results.")

    # initialise things
    exp_dict = {}
    base_dict = {}
    map_per_it = []

    # load data
    file = np.load(args['load_loc'], allow_pickle=True)
    results = file['results'].item()

    # generate histogram for each iteration of interest
    its = [2,5,10,20,30,50]
    for it_of_interest in its:
        key = args['save_name'] + f'_iteration_{it_of_interest}'
        per_species_AP = results[it_of_interest]['per_species_average_precision']
        exp_dict[key] = per_species_AP

    # Use seaborn's whitegrid theme
    sns.set_theme(style="whitegrid")
    # Use serif font
    plt.rcParams['font.family'] = 'serif'

    # Define the maximum y-value for all histograms
    max_freq = max([np.histogram(list(value.values()), bins=50)[0].max() for value in exp_dict.values()])

    num_bins = 50
    bins = np.linspace(0, 1, num_bins)
    for i, (key, value) in enumerate(exp_dict.items()):

        # Create histogram using seaborn
        sns.histplot(data=value, bins=bins, color='dodgerblue', kde=False)
        # Add title and labels
        plt.title(f'Timestep {its[i]}')
        plt.xlabel('Average Precision')
        plt.ylabel('Frequency')

        # Set consistent y-axis
        plt.ylim(0, max_freq)
        plt.xlim(0, 1)

        # Show plot
        plt.tight_layout()

        save_loc_pdf = os.path.join(args['save_dir'], (key + ".pdf"))
        save_loc_png = os.path.join(args['save_dir'], (key + ".png"))

        # Save figure
        plt.savefig(save_loc_pdf)

        plt.close()

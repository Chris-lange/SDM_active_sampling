import argparse
import sys
sys.path.append('../')
from active_sampling import active_sampler

# visualizes predictions made by trained active sampling model and plots sampled data

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_loc', type=str, default="saved_figures/figure_name")
    parser.add_argument('--load_loc', type=str, default="../results/WA_HSS_IUCN_1.npz")
    parser.add_argument('--title', type=str, default="")
    parser.add_argument('--taxa_of_interest', type=int, default=17090) # Species to visualize
    parser.add_argument('--iteration_of_interest', type=int, default=50) # Which step in the process you wish to visualize
    parser.add_argument('--plot_ground_truth', type=bool, default=False) # Whether to show the "ground truth" presence data
    parser.add_argument('--plot_samples', type=bool, default=True) # Whether to show the locations of sampled points
    parser.add_argument('--show_average_precision', type=bool, default=True) # Give the Average Precision for this species and time step

    args = vars(parser.parse_args())

    active_sampler = active_sampler.ActiveSampler.from_results_file(results_file_path=args['load_loc'], iteration=args['iteration_of_interest'], save=False)

    # Or plot_samples_and_probabilities_no_mask to generate predictions over the ocean too
    # Or plot_base_model_probs to generate predictions for species from the hypothesis set
    active_sampler.plot_samples_and_probabilities(taxa=args['taxa_of_interest'], save_loc=args['save_loc'],
                                                  give_probs=args['show_average_precision'],
                                                  show_gt=args['plot_ground_truth'],
                                                  show_samples=args['plot_samples'])



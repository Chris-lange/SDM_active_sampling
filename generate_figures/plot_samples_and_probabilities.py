import argparse
from active_sampling import active_sampler

# visualises predictions made by trained active sampling model and plots sampled data
# need to add a way to choose iteration

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_loc', type=str, default="saved_figures/test_plot_gt")
    parser.add_argument('--load_loc', type=str, default="../results/all_results/WA_HSS_IUCN_1.npz")
    parser.add_argument('--title', type=str, default="")
    parser.add_argument('--taxa_of_interest', type=int, default=17090)
    parser.add_argument('--iteration_of_interest', type=int, default=50)
    parser.add_argument('--plot_ground_truth', type=bool, default=True)
    parser.add_argument('--plot_samples', type=bool, default=False)
    parser.add_argument('--show_average_precision', type=bool, default=False)

    args = vars(parser.parse_args())

    active_sampler = active_sampler.ActiveSampler.from_results_file(results_file_path=args['load_loc'], iteration=args['iteration_of_interest'], save=False)

    active_sampler.plot_samples_and_probabilities(taxa=args['taxa_of_interest'], save_loc=args['save_loc'],
                                                  give_probs=args['show_average_precision'],
                                                  show_gt=args['plot_ground_truth'],
                                                  show_samples=args['plot_samples'])



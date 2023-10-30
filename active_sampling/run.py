from active_sampler import ActiveSampler
import argparse
import torch
import os
import random
import numpy as np
import json

if __name__ == '__main__':

    # check for GPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load JSON config
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Argument parser
    parser = argparse.ArgumentParser()
    # ~#~#~#~#~#~#~#~ General Arguments ~#~#~#~#~#~#~#~#
    # parser.add_argument('--experiment', type=str, required=True,    # Automatically sets seed, eval_type, init data, etc
    #                     choices=['IUCN_1', 'IUCN_2', 'IUCN_3', 'SNT_1', 'SNT_2', 'SNT_3'])
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--save_name', type=str, default='DEFAULT_NAME')  # file name for results

    # ~#~#~#~#~#~#~#~ Active Sampling Arguments ~#~#~#~#~#~#~#~#
    parser.add_argument('--steps', type=int, default=50)            # number of active sampling steps to complete

    parser.add_argument('--update_method', type=str, default='weighted_average_plus',    # updating strategy
                        choices=['logistic_regression',         # logistic regression
                                 'weighted_average',            # weighted averaging of hypotheses
                                 'weighted_average_plus'])      # weighted averaging of hypotheses with additional LR classifier

    parser.add_argument('--sampling_method', type=str, default='query_by_HS_committee',  # sampling strategy
                        choices=['random_sampling',             # locations sampled at random
                                 'most_positive_pred',          # location with most positive model prediction sampled
                                 'most_uncertain_pred',         # location with model prediction closest to 0.5 sampled
                                 'largest_expected_model_change',# location that has largest expected model change sampled
                                 'query_by_HS_committee',       # location that has most disagreement between weighted committee of hypotheses sampled
                                 'query_by_committee'])         # "traditional" query by committee (committee members trained on subsets of data)

    parser.add_argument('--detectability', type=float,
                        default=1.0)                                # probability of spotting species given it is present

    # ~#~#~#~#~#~#~#~ WA_HSS and WA_HSS+ Arguments ~#~#~#~#~#~#~#~#
    parser.add_argument('--probability_combination_method', type=str,   # method of calculating each hypothesis's agreement with the data to determine hypothesis weights
                        default='product', choices=['product',  # standard way of calculating joint probability assuming independence
                                            'average'])  # mean of probabilities

    parser.add_argument('--member_weighting', type=str,   # for determining contribution of each committee members votes for location to sample
                        default='agreement', choices=['agreement',  # weight by agreement with data
                                            'equal'])  # no weighting, consider all committee members equally

    parser.add_argument('--vote_type', type=str,   # Use soft votes or hard votes
                        default='soft', choices=['soft',  # weight votes by committee member's uncertainty
                                            'hard'])

    parser.add_argument('--max_committee_size', type=int,
                        default=100)  # maximum number of committee members for query by committee, -1 to use all (slow)

    args = parser.parse_args()

    args = vars(args)

    # Loop through all experiment types
    for experiment in ['IUCN_1', 'IUCN_2', 'IUCN_3', 'SNT_1', 'SNT_2', 'SNT_3']:
        # Set the experiment type
        args['experiment'] = experiment

        # Extract dataset and run number
        dataset, run_number = args['experiment'].split('_')
        run_number = int(run_number) - 1  # Convert to zero-based index

        # Set per experiment arguments
        args['model_path'] = config['model_path']
        args['env_loc'] = config['env_loc']
        args['save_dir'] = config['save_dir']
        args['ocean_mask_loc'] = config['ocean_mask_loc']
        args['init_data_loc'] = config['init_data_loc'][dataset][run_number]
        args['taxa_list_loc'] = config['taxa_list_loc'][dataset]
        args['gt_data_loc'] = config['gt_data'][dataset]
        args['eval_type'] = config['eval_type'][dataset]
        args['seed'] = config['seeds'][run_number]
        args['batch_size'] = config['batch_size']
        args['use_linear_model'] = config['use_linear_model']


        # Make directories
        if not os.path.isdir(args['save_dir']):
            os.makedirs(args['save_dir'])

        # Set the random seeds
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])
        random.seed(args['seed'])

        # Create and run the active sampler
        sampler = ActiveSampler(args=args)

        for i in range(args['steps']):
            sampler.iterate()
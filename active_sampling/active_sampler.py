import copy
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from active_sampling.active_sampling_utils import mask_to_locations
from active_sampling.update import UpdateStrategy
from active_sampling.sample import  SamplingStrategy
from active_sampling.evaluate import ModelEvaluator
from active_sampling.data_management import DataManager
from backbone import backbone_utils

class ActiveSampler:
    def __init__(self, args, iteration=None, loaded_from_results=False):

        self.args = args
        self.loaded_from_results = loaded_from_results

        # Initialise data
        self.data_manager = DataManager(args, iteration)

        # Model-related initializations
        self.base_model, self.backbone_params = backbone_utils.load_model(args)
        self.active_sample_model = copy.deepcopy(self.base_model)
        self.base_model_taxa = self.backbone_params['class_to_taxa']
        self.active_sample_model, self.backbone_params = backbone_utils.modify_model(
            self.active_sample_model, self.backbone_params, self.data_manager.taxa_map
        )

        # generate backbone features
        self.data_manager.generate_locs_to_feats(backbone_params=self.backbone_params, base_model=self.base_model)

        # Active Sampling related initializations
        self.updater = UpdateStrategy(args=self.args, data_manager=self.data_manager)
        self.sampler = SamplingStrategy(args=self.args, data_manager=self.data_manager)
        self.evaluator = ModelEvaluator(args, self.data_manager)

        # keep track of steps (-1 as initial update yet to happen)
        self.update_steps = -1
        self.sample_steps = 0

        # Additional initializations based on update type
        if self.args['update_method'] == 'weighted_average_plus':
            self.additional_classifiers = copy.deepcopy(self.active_sample_model)
        else:
            self.additional_classifiers = None

        # Perform an initial update to create classifiers from initial data
        self.update()

        # Perform an initial eval, so we have results from step 0 (if not loading from results file)
        if self.loaded_from_results == False:
            self.eval()
            # pass

    def update(self):
        self.active_sample_model, self.additional_classifiers = self.updater.update(
            model=self.active_sample_model,
            base_model=self.base_model,
            additional_classifiers=self.additional_classifiers)
        self.update_steps += 1
        return self.active_sample_model, self.additional_classifiers

    def sample(self):
        # Generate samples using chosen sampling strategy
        samples = self.sampler.sample(model=self.active_sample_model, base_model=self.base_model,
                                      additional_classifiers=self.additional_classifiers)

        # Label the sampled data
        self.label(locations=samples)

        # Update the sampling count
        self.sample_steps += 1

    def eval(self):
        self.evaluator.eval(
            model=self.active_sample_model, update_steps=self.update_steps
        )
        # save results (unless we have loaded the sampler from an existing results file)
        # could add a little more here to save if the iteration that is being evaled doesn't yet exist in results
        if self.loaded_from_results == False:
            self.save_results()
        return

    def label(self, locations):
        # Get detectability setting from arguments
        detectability = self.args['detectability']

        # Initialize dictionary to store the labels
        label_dict = {}

        # Iterate through each location by taxon_id
        for taxon_id, loc_dict in locations.items():

            cell_index = loc_dict["idx"]
            cell_centre = loc_dict["location"]

            # Generate a random probability to simulate detection
            detect_prob = np.random.random()

            # Check if the cell is present in ground truth data for the given taxon
            if cell_index in self.data_manager.gt_data['taxa_data'][taxon_id]['presence']:
                detect = 1  # Positive detection
                # Update label based on detectability
                if detect_prob < detectability:
                    # ground truth positive and species detected
                    label_dict[taxon_id] = (cell_centre, 1, 1, cell_index)
                else:
                    # ground truth positive and species not detected
                    label_dict[taxon_id] = (cell_centre, 0, 1, cell_index)
            else:
                # species not present
                detect = 0
                label_dict[taxon_id] = (cell_centre, 0, 0, cell_index)

            # Print diagnostic information
            print(
                f'Searched cell index: {cell_index}, location: {cell_centre} for taxon_id {taxon_id}')
            print(f'Ground truth is {"positive" if detect == 1 else "negative"}')

        # Update the data manager with the new labels
        self.data_manager.update_data_lists(label_dict)

        return label_dict

    def save_results(self):
        if self.args['save_name'] != '':
            save_name = os.path.join(self.args['save_dir'], (self.args['save_name'] + f"{self.args['experiment']}" ))
            print('Saving results and sampled data...')
            np.savez(file=save_name, results=self.evaluator.results, data=self.data_manager.data_lists,
                     args=self.args, params=self.backbone_params, sample_steps=self.sample_steps,
                     update_steps=self.update_steps)

    def save_final_model(self, model, save_loc):
        print(f'Saving output model to {save_loc}')
        op_state = {'epoch': 10,
                    'state_dict': model.state_dict(),
                    'params': self.backbone_params,
                    'args': self.args}
        torch.save(op_state, save_loc)

    def iterate(self):
        # Performs a full active sampling loop
        self.sample()
        self.update()
        self.eval()

    @classmethod
    def from_results_file(cls, results_file_path, model_path_override=None,
                          gt_path_override=None, device_override=None, iteration=None, save=False):

        # Load existing results file
        loaded_data = np.load(results_file_path, allow_pickle=True)
        loaded_args = loaded_data['args'].item()
        loaded_sample_steps = loaded_data.get('sample_steps', 0)  # Default to 0 if not found
        loaded_update_steps = loaded_data.get('update_steps', -1)  # Default to -1 if not found

        # Override the model path if specified
        if model_path_override:
            loaded_args['model_path'] = model_path_override

        if gt_path_override:
            loaded_args['gt_data_loc'] = gt_path_override

        if device_override:
            loaded_args['device'] = device_override

        # Initialize Active_Sampler with loaded data
        loaded_args['init_data_loc'] = results_file_path
        if iteration is None:
            instance = cls(loaded_args, loaded_from_results=True)
        else:
            instance = cls(loaded_args, iteration=iteration, loaded_from_results=True)

        # Set the loaded sample and update steps to the instance
        instance.sample_steps = loaded_sample_steps
        instance.update_steps = loaded_update_steps

        # Set the saved results to the evaluator
        instance.evaluator.results = loaded_data['results'].item()

        return instance

    def plot_samples_and_probabilities(self, taxa, save_loc, give_probs=False, show_gt=False, show_samples=False):
        # Plot a world map using ocean mask, and overlay model predictions and sampled locations.

        sampled_locs = self.data_manager.data_lists[taxa]['data']

        # Here we are showing the discovered label for each sample, not the ground truth
        labels = self.data_manager.data_lists[taxa]['labels']
        labels = labels[:, :1].flatten() # labels[:, :0] for ground truth labels

        mask_loc = self.args['ocean_mask_loc']
        mask = np.load(file=mask_loc, allow_pickle=True)

        # get locations corresponding to points on the mask
        scaled_mask_locs, loc_to_mask_dict = mask_to_locations(mask=mask, lat_range=(-1, 1), lon_range=(-1, 1))


        # get context_feats if they exist
        if self.backbone_params['use_context_feats']:
            context_feats = backbone_utils.load_context_feats(self.args['env_loc'], self.backbone_params['device'])
        else:
            context_feats = None

        if show_gt==False:
            # Convert numpy array to PyTorch tensor
            scaled_mask_locs = torch.tensor(scaled_mask_locs, dtype=torch.float32).to(self.args['device'])

            # Get model predictions
            num_locs = len(scaled_mask_locs)
            preds = []

            # Generate predictions in batches to avoid memory issues
            for i in range(0, num_locs, self.args['batch_size']):
                end = min(i + self.args['batch_size'], num_locs)
                batch_locs = scaled_mask_locs[i:end]
                with torch.no_grad():

                    batch_input_feats = backbone_utils.generate_input_feats(batch_locs, 0, params=self.backbone_params,
                                                                            context_feats=context_feats,
                                                                            device=self.args["device"])

                    batch_preds = self.active_sample_model(batch_input_feats, class_of_interest=self.data_manager.taxa_map[taxa],
                                        use_feats_as_input=False)

                    # preds.append(batch_preds.cpu().to_numpy()))

                    list_batch_preds = (batch_preds.cpu()).tolist()
                    preds.extend(list_batch_preds)

            # Create an empty array filled with "bad" values (np.nan)
            pred_map = np.full(mask.shape, np.nan)
            scaled_mask_locs = np.float32(scaled_mask_locs)
            for idx, loc in enumerate(scaled_mask_locs):
                j, i = loc_to_mask_dict[tuple(loc)]
                pred = preds[idx]
                pred_map[i, j] = pred

            # Create a custom colormap with a transparent color for "bad" values
            my_cmap = plt.get_cmap('plasma').copy()  # Changed to 'plasma'
            my_cmap.set_bad(color='none')

            # Plot the heatmap
            plt.figure(figsize=(40.08, 20.04), dpi=100)
            plt.imshow(pred_map, cmap=my_cmap, interpolation='nearest', extent=(-180, 180, -90, 90))
            # plt.colorbar()
            if give_probs:
                average_precision = self.evaluator.results[len(sampled_locs)-2]['per_species_average_precision'][taxa]
                plt.title(f'Taxon: {taxa}, {len(sampled_locs)} samples, Average Precision: {average_precision}')

            # Remove axes and tick marks
            plt.axis('off')

            # Plot sampled locations with labels
            if len(sampled_locs) == len(labels):  # Make sure labels exist and are of the same length
                if show_samples:
                    for loc, label in zip(sampled_locs, labels):
                        color = '#18A803' if label == 1 else '#FC1109'
                        plt.scatter(loc[0], loc[1], c=color, marker='x', linewidths=11, s=500)

        else:
            taxa_presence_inds = list(self.data_manager.gt_data['taxa_data'][taxa]['presence'])
            taxa_presence_locs = self.data_manager.gt_data['locs'][taxa_presence_inds]
            plt.figure(figsize=(40.08, 20.04), dpi=100)
            plt.imshow(mask, extent=(-180, 180, -90, 90), interpolation='nearest', cmap='Blues')
            for i, loc in enumerate(taxa_presence_locs):
                marker = 'o'
                color = (0.940015, 0.975158, 0.131326, 1.0)
                plt.plot(loc[0], loc[1], marker=marker, color=color, markersize=0.5)
                plt.axis('off')

        plt.tight_layout()
        pdf_save_loc = save_loc + '.pdf'
        png_save_loc = save_loc + '.png'
        plt.savefig(pdf_save_loc, dpi=100)
        plt.savefig(png_save_loc, dpi=100)


    def plot_samples_and_probabilities_no_mask(self, taxa, save_loc, give_probs=False, show_gt=False, show_samples=False):
        # Plot a world map using ocean mask, and overlay model predictions and sampled locations.

        sampled_locs = self.data_manager.data_lists[taxa]['data']

        # Here we are showing the discovered label for each sample, not the ground truth
        labels = self.data_manager.data_lists[taxa]['labels']
        labels = labels[:, :1].flatten() # labels[:, :0] for ground truth labels

        mask_loc = self.args['ocean_mask_loc']
        mask = np.load(file=mask_loc, allow_pickle=True)
        mask = np.ones(mask.shape)

        # get locations corresponding to points on the mask
        scaled_mask_locs, loc_to_mask_dict = mask_to_locations(mask=mask, lat_range=(-1, 1), lon_range=(-1, 1))


        # get context_feats if they exist
        if self.backbone_params['use_context_feats']:
            context_feats = backbone_utils.load_context_feats(self.args['env_loc'], self.backbone_params['device'])
        else:
            context_feats = None

        if show_gt==False:
            # Convert numpy array to PyTorch tensor
            scaled_mask_locs = torch.tensor(scaled_mask_locs, dtype=torch.float32).to(self.args['device'])

            # Get model predictions
            num_locs = len(scaled_mask_locs)
            preds = []

            # Generate predictions in batches to avoid memory issues
            for i in range(0, num_locs, self.args['batch_size']):
                end = min(i + self.args['batch_size'], num_locs)
                batch_locs = scaled_mask_locs[i:end]
                with torch.no_grad():

                    batch_input_feats = backbone_utils.generate_input_feats(batch_locs, 0, params=self.backbone_params,
                                                                            context_feats=context_feats,
                                                                            device=self.args["device"])

                    batch_preds = self.active_sample_model(batch_input_feats, class_of_interest=self.data_manager.taxa_map[taxa],
                                        use_feats_as_input=False)

                    # preds.append(batch_preds.cpu().to_numpy()))

                    list_batch_preds = (batch_preds.cpu()).tolist()
                    preds.extend(list_batch_preds)

            # Create an empty array filled with "bad" values (np.nan)
            pred_map = np.full(mask.shape, np.nan)
            scaled_mask_locs = np.float32(scaled_mask_locs)
            for idx, loc in enumerate(scaled_mask_locs):
                j, i = loc_to_mask_dict[tuple(loc)]
                pred = preds[idx]
                pred_map[i, j] = pred

            # Create a custom colormap with a transparent color for "bad" values
            my_cmap = plt.get_cmap('plasma').copy()  # Changed to 'plasma'
            my_cmap.set_bad(color='none')

            # Plot the heatmap
            plt.figure(figsize=(40.08, 20.04), dpi=100)
            plt.imshow(pred_map, cmap=my_cmap, interpolation='nearest', extent=(-180, 180, -90, 90))
            # plt.colorbar()
            if give_probs:
                average_precision = self.evaluator.results[len(sampled_locs)-2]['per_species_average_precision'][taxa]
                plt.title(f'Taxon: {taxa}, {len(sampled_locs)} samples, Average Precision: {average_precision}')

            # Remove axes and tick marks
            plt.axis('off')

            # Plot sampled locations with labels
            if len(sampled_locs) == len(labels):  # Make sure labels exist and are of the same length
                if show_samples:
                    for loc, label in zip(sampled_locs, labels):
                        color = '#18A803' if label == 1 else '#FC1109'
                        plt.scatter(loc[0], loc[1], c=color, marker='x', linewidths=11, s=500)

        else:
            taxa_presence_inds = list(self.data_manager.gt_data['taxa_data'][taxa]['presence'])
            taxa_presence_locs = self.data_manager.gt_data['locs'][taxa_presence_inds]
            plt.figure(figsize=(40.08, 20.04), dpi=100)
            plt.imshow(mask, extent=(-180, 180, -90, 90), interpolation='nearest', cmap='Blues')
            for i, loc in enumerate(taxa_presence_locs):
                marker = 'o'
                color = (0.940015, 0.975158, 0.131326, 1.0)
                plt.plot(loc[0], loc[1], marker=marker, color=color, markersize=2)
                plt.axis('off')

        plt.tight_layout()
        pdf_save_loc = save_loc + '.pdf'
        png_save_loc = save_loc + '.png'
        plt.savefig(pdf_save_loc, dpi=100)
        plt.savefig(png_save_loc, dpi=100)


    def plot_base_model_probs(self, base_model_taxa, save_loc):
        # Plot a world map using ocean mask, and overlay model predictions and sampled locations.

        mask_loc = self.args['ocean_mask_loc']
        mask = np.load(file=mask_loc, allow_pickle=True)

        # get locations corresponding to points on the mask
        scaled_mask_locs, loc_to_mask_dict = mask_to_locations(mask=mask, lat_range=(-1, 1), lon_range=(-1, 1))

        # get context_feats if they exist
        if self.backbone_params['use_context_feats']:
            context_feats = backbone_utils.load_context_feats(self.args['env_loc'], self.backbone_params['device'])
        else:
            context_feats = None

        # Convert numpy array to PyTorch tensor
        scaled_mask_locs = torch.tensor(scaled_mask_locs, dtype=torch.float32).to(self.args['device'])

        # Get model predictions
        num_locs = len(scaled_mask_locs)
        preds = []

        # Generate predictions in batches to avoid memory issues
        for i in range(0, num_locs, self.args['batch_size']):
            end = min(i + self.args['batch_size'], num_locs)
            batch_locs = scaled_mask_locs[i:end]
            with torch.no_grad():

                batch_input_feats = backbone_utils.generate_input_feats(batch_locs, 0, params=self.backbone_params,
                                                                        context_feats=context_feats,
                                                                        device=self.args["device"])

                batch_preds = self.base_model(batch_input_feats, class_of_interest=self.base_model_taxa.index(base_model_taxa),
                                    use_feats_as_input=False)

                # preds.append(batch_preds.cpu().to_numpy()))

                list_batch_preds = (batch_preds.cpu()).tolist()
                preds.extend(list_batch_preds)

        # Create an empty array filled with "bad" values (np.nan)
        pred_map = np.full(mask.shape, np.nan)
        scaled_mask_locs = np.float32(scaled_mask_locs)
        for idx, loc in enumerate(scaled_mask_locs):
            j, i = loc_to_mask_dict[tuple(loc)]
            pred = preds[idx]
            pred_map[i, j] = pred

        # Fill in the predicted values at land locations
        land_indices = np.argwhere(mask == 1)
        #flat_preds = torch.cat(preds).numpy()


        for idx, (i, j) in enumerate(land_indices):
            pred_map[i, j] = preds[idx]

        # Create a custom colormap with a transparent color for "bad" values
        my_cmap = plt.get_cmap('plasma').copy()  # Changed to 'plasma'
        my_cmap.set_bad(color='none')

        # Plot the heatmap
        plt.figure(figsize=(40.08, 20.04), dpi=100)
        plt.imshow(pred_map, cmap=my_cmap, interpolation='nearest', extent=(-180, 180, -90, 90))
        # plt.colorbar()

        # Remove axes and tick marks
        plt.axis('off')

        plt.tight_layout()
        pdf_save_loc = save_loc + '.pdf'
        png_save_loc = save_loc + '.png'
        plt.savefig(pdf_save_loc, dpi=100)
        plt.savefig(png_save_loc, dpi=100)
        plt.close()
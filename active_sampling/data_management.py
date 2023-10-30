import json
import random
import sys
import numpy as np
sys.path.append('../')
from backbone import backbone_utils

class DataManager:
    def __init__(self, args, iteration=None):
        self.args = args
        self.taxa_map = self.generate_taxa_map()
        self.data_lists = self.load_initial_data_list()
        if iteration is not None:
            self.set_data_list_iteration(iteration=iteration)
        if self.args['eval_type'] == 'IUCN':
            self.gt_data = self.create_IUCN_ground_truth()
        elif self.args['eval_type'] == 'SNT':
            self.gt_data = self.create_snt_ground_truth()
        else:
            print("ground truth dataset not recognised")
            raise NotImplementedError
        # init locs_to_feats
        # needs to be updated later once model is determined
        self.locs_to_feats = None

    def generate_taxa_map(self):
        # load taxa list
        taxa = list(np.load(self.args['taxa_list_loc'], allow_pickle=True))

        # map between taxa and class index
        taxa_map = {}
        for i, taxon_id in enumerate(taxa):
            taxa_map[int(taxon_id)] = i
        return taxa_map

    def load_initial_data_list(self):
        init_data = np.load(self.args['init_data_loc'], allow_pickle=True)
        if 'data' in init_data:
            data_list = init_data['data'].item()
            for taxon in self.taxa_map.keys():
                data_list[taxon]['class_idx'] = self.taxa_map[taxon]
            return data_list
        else:
            raise KeyError("The required 'data' field is missing in the initial data file.")

    def update_data_lists(self, label_dict={}):
        # add newly sampled data to the sampled data lists
        for taxon_id, value in label_dict.items():
            old_data = self.data_lists[taxon_id]['data']
            old_labs = self.data_lists[taxon_id]['labels']
            old_inds = self.data_lists[taxon_id]['sampled_idx']
            new_data = np.vstack((old_data, value[0]))
            new_labs = np.vstack((old_labs, np.array([value[1], value[2]])))
            new_idx = np.vstack((old_inds, value[3]))
            self.data_lists[taxon_id]['data'] = new_data
            self.data_lists[taxon_id]['labels'] = new_labs
            self.data_lists[taxon_id]['sampled_idx'] = new_idx

    def create_snt_ground_truth(self):
        # Load ground truth data
        snt_data_loc = self.args['gt_data_loc']
        D = np.load(snt_data_loc, allow_pickle=True).item()

        # Create a mask to filter out ocean points
        loc_mask = self.get_mask_as_input_numpy(D['obs_locs'])
        bool_mask = loc_mask != 0.0

        # Apply the mask to filter out ocean points
        filtered_obs_locs = D['obs_locs'][bool_mask]

        # Initialize filtered taxa data dictionary
        filtered_taxa_data = {}
        count = 0

        # Create a mapping between original indices and filtered indices
        orig_to_filtered_idx_map = {orig_idx: filtered_idx for filtered_idx, orig_idx in
                                    enumerate(np.where(bool_mask)[0])}

        # Iterate over taxa, location indices, and labels
        for taxon, indices, labels in zip(D['taxa'], D['loc_indices_per_species'], D['labels_per_species']):
            # Only consider taxa present in the taxa_map
            if taxon in self.taxa_map:
                # Generate presence and absence indices, filtered by the mask
                presence_indices = np.array(
                    [orig_to_filtered_idx_map[idx] for idx, label in zip(indices, labels) if
                     label == 1 and idx in orig_to_filtered_idx_map],
                    dtype=np.int32)

                absence_indices = np.array(
                    [orig_to_filtered_idx_map[idx] for idx, label in zip(indices, labels) if
                     label == 0 and idx in orig_to_filtered_idx_map],
                    dtype=np.int32)
                # Update the filtered taxa data
                filtered_taxa_data[int(taxon)] = {'presence': presence_indices, 'absence': absence_indices}
                count += 1
                num_present = len(presence_indices)
                num_absent = len(absence_indices)
                print(f"{taxon}, {count}, present cells: {num_present}, absent_cells: {num_absent}")

        # Return the final ground truth data
        return {
            'meta_data': {'resolution': 5},
            'taxa_data': filtered_taxa_data,
            'locs': np.array(filtered_obs_locs, dtype=np.float32)
        }

    def create_IUCN_ground_truth(self):
        # Load IUCN data from a JSON file
        with open(self.args['gt_data_loc'], 'r') as f:
            iucn_data = json.load(f)

        # Create a mask to filter out ocean points
        loc_mask = self.get_mask_as_input_numpy(np.array(iucn_data['locs']))
        bool_mask = loc_mask != 0.0

        # Apply the mask to filter out ocean points
        filtered_obs_locs = np.array(iucn_data['locs'])[bool_mask]

        # Create a mapping between original indices and filtered indices
        orig_to_filtered_idx_map = {orig_idx: filtered_idx for filtered_idx, orig_idx in
                                    enumerate(np.where(bool_mask)[0])}

        # Initialize set of all cell indices
        all_cells = set(range(len(filtered_obs_locs)))

        # Initialize filtered taxa data dictionary
        filtered_taxa_data = {}

        # Iterate over each taxon and its presence indices
        count = 0
        for taxon, presence_indices in iucn_data['taxa_presence'].items():
            # Only consider taxa present in the taxa_map
            if int(taxon) in self.taxa_map.keys():
                # Generate presence indices, filtered by the mask and mapped to new indices
                presence_indices = np.array(
                    [orig_to_filtered_idx_map[idx] for idx in presence_indices if idx in orig_to_filtered_idx_map],
                    dtype=np.int32)

                # Generate absence indices by subtracting presence indices from all cells
                absence_indices = np.array(list(all_cells - set(presence_indices)), dtype=np.int32)

                # Update the filtered taxa data
                filtered_taxa_data[int(taxon)] = {'presence': presence_indices, 'absence': absence_indices}
                if len(presence_indices) < 1:
                    print(f"{taxon}, no presence!")
                elif len(absence_indices) < 1:
                    print(f"{taxon}, no absence!")
                else:
                    num_present = len(presence_indices)
                    count += 1
                    print(f"{taxon}, {count}, present cells: {num_present}")

        if count != 500:
            print(f"ground truth only generated for {count} / 500 species!")
        else:
            print(f"ground truth generated for all 500 species!")
        # Return the final ground truth data
        return {
            'meta_data': {'resolution': iucn_data['meta_data']['resolution']},
            'taxa_data': filtered_taxa_data,
            'locs': np.array(filtered_obs_locs, dtype=np.float32)
        }

    def get_mask_as_input_numpy(self, locs):
        # load mask
        mask_loc = self.args['ocean_mask_loc']
        mask = np.load(file=mask_loc, allow_pickle=True)
        # Scale locations to [0, 1] and adjust latitude
        scaled_locs = (locs + np.array([180, 90])) / np.array([360, 180])
        scaled_locs[:, 1] = 1 - scaled_locs[:, 1]

        # Scale to data size
        scaled_locs[:, 0] *= (mask.shape[1] - 1)
        scaled_locs[:, 1] *= (mask.shape[0] - 1)

        # Get integer coordinates
        loc_int = np.floor(scaled_locs).astype(int)
        xx, yy = loc_int[:, 0], loc_int[:, 1]

        # Compute deltas for interpolation
        loc_delta = scaled_locs - loc_int
        dx, dy = loc_delta[:, 0], loc_delta[:, 1]

        # Ensure we don't go out of bounds
        xx_plus = np.clip(xx + 1, 0, mask.shape[1] - 1)
        yy_plus = np.clip(yy + 1, 0, mask.shape[0] - 1)

        # Bilinear interpolation
        interp_val = (mask[yy, xx] * (1 - dx) * (1 - dy) +
                      mask[yy, xx_plus] * dx * (1 - dy) +
                      mask[yy_plus, xx] * (1 - dx) * dy +
                      mask[yy_plus, xx_plus] * dx * dy)

        return interp_val

    def generate_locs_to_feats(self, backbone_params, base_model):
        # Load backbone features
        if backbone_params['use_context_feats']:
            context_feats = backbone_utils.load_context_feats(self.args['env_loc'], backbone_params['device'])
        else:
            context_feats = None
        locs_to_feats = backbone_utils.generate_feats_array(
            model=base_model, params=backbone_params, args=self.args,
            context_feats=context_feats, eval_data=self.gt_data
        )
        self.locs_to_feats = locs_to_feats
        return

    def get_sampled_feats_and_labels(self, taxon_id):
        # Get data and labels for the specific taxon
        data_for_taxon = self.data_lists[taxon_id]
        data = data_for_taxon['data']
        labels = data_for_taxon['labels'][:, 0]
        sampled_idx = data_for_taxon['sampled_idx']

        # Get positive and negative locations and their corresponding indices
        pos_locs = data[labels == 1]
        neg_locs = data[labels == 0]
        pos_inds = sampled_idx[labels == 1].flatten()
        neg_inds = sampled_idx[labels == 0].flatten()

        # Combine positive and negative locations and indices
        all_locs = np.vstack((pos_locs, neg_locs))
        all_inds = np.array([*pos_inds, *neg_inds])  # Convert to 1D array directly

        # Create label array
        length_pos = len(pos_locs)
        labels = np.ones(all_locs.shape[0], dtype=np.int32)
        labels[length_pos:] = 0

        # Fetch features based on indices
        all_feats = self.locs_to_feats[all_inds]

        return all_feats, labels, length_pos

    def set_data(self, new_data):
        self.data = new_data

    def create_IUCN_init_data(self, save_loc):
        # Load IUCN data from a JSON file
        with open(self.args['gt_data_loc'], 'r') as f:
            iucn_data = json.load(f)

        # Create a mask to filter out ocean points
        loc_mask = self.get_mask_as_input_numpy(np.array(iucn_data['locs']))
        bool_mask = loc_mask != 0.0

        # Apply the mask to filter out ocean points
        filtered_obs_locs = np.array(iucn_data['locs'])[bool_mask]

        # Create a mapping between original indices and filtered indices
        orig_to_filtered_idx_map = {orig_idx: filtered_idx for filtered_idx, orig_idx in
                                    enumerate(np.where(bool_mask)[0])}

        # Initialize set of all cell indices
        all_cells = set(range(len(filtered_obs_locs)))

        for i in range(3):

            # Initialize filtered taxa data dictionary
            filtered_taxa_data = {}
            data = {}
            # Iterate over each taxon and its presence indices
            count = 0
            for taxon, presence_indices in iucn_data['taxa_presence'].items():
                # Only consider taxa present in the taxa_map
                if int(taxon) in self.taxa_map.keys():
                    # Generate presence indices, filtered by the mask and mapped to new indices
                    presence_indices = np.array(
                        [orig_to_filtered_idx_map[idx] for idx in presence_indices if idx in orig_to_filtered_idx_map],
                        dtype=np.int32)

                    # Generate absence indices by subtracting presence indices from all cells
                    absence_indices = np.array(list(all_cells - set(presence_indices)), dtype=np.int32)

                    # Update the filtered taxa data
                    filtered_taxa_data[int(taxon)] = {'presence': presence_indices, 'absence': absence_indices}
                    if len(presence_indices) < 1:
                        print(f"{taxon}, no presence!")
                    elif len(absence_indices) < 1:
                        print(f"{taxon}, no absence!")
                    else:
                        num_present = len(presence_indices)
                        count += 1
                        print(f"{taxon}, {count}, present cells: {num_present}")

                        init_pos_inds = random.sample(list(presence_indices), 1)
                        init_neg_inds = random.sample(list(absence_indices), 1)
                        sampled_idx = np.array([init_pos_inds, init_neg_inds])
                        init_pos_loc = np.float32(filtered_obs_locs[init_pos_inds])
                        init_neg_loc = np.float32(filtered_obs_locs[init_neg_inds])
                        locs = np.vstack((init_pos_loc, init_neg_loc))
                        idx = count - 1
                        pos_labs = np.array([1,1])
                        neg_labs = np.array([0,0])
                        labels = np.vstack((pos_labs,neg_labs))
                        data[int(taxon)] = {}
                        data[int(taxon)]['data'] = locs
                        data[int(taxon)]['class_idx'] = idx
                        data[int(taxon)]['sampled_idx'] = sampled_idx
                        data[int(taxon)]['labels'] = labels

            save_name = f"{save_loc}_{i+1}"
            np.savez(data=data, file=save_name)
        return


    def create_snt_init_data(self, save_loc):
        # Load ground truth data
        snt_data_loc = self.args['gt_data_loc']
        D = np.load(snt_data_loc, allow_pickle=True).item()

        # Create a mask to filter out ocean points
        loc_mask = self.get_mask_as_input_numpy(D['obs_locs'])
        bool_mask = loc_mask != 0.0

        # Apply the mask to filter out ocean points
        filtered_obs_locs = D['obs_locs'][bool_mask]

        # Create a mapping between original indices and filtered indices
        orig_to_filtered_idx_map = {orig_idx: filtered_idx for filtered_idx, orig_idx in
                                    enumerate(np.where(bool_mask)[0])}

        for i in range(3):

            # Initialize filtered taxa data dictionary
            filtered_taxa_data = {}
            data = {}
            # Iterate over each taxon and its presence indices
            count = 0
            # Iterate over taxa, location indices, and labels
            for taxon, indices, labels in zip(D['taxa'], D['loc_indices_per_species'], D['labels_per_species']):
                # Only consider taxa present in the taxa_map
                if taxon in self.taxa_map:
                    # Generate presence and absence indices, filtered by the mask
                    presence_indices = np.array(
                        [orig_to_filtered_idx_map[idx] for idx, label in zip(indices, labels) if label == 1 and idx in orig_to_filtered_idx_map],
                        dtype=np.int32)

                    absence_indices = np.array(
                        [orig_to_filtered_idx_map[idx] for idx, label in zip(indices, labels) if label == 0 and idx in orig_to_filtered_idx_map],
                        dtype=np.int32)

                    # Update the filtered taxa data
                    filtered_taxa_data[int(taxon)] = {'presence': presence_indices, 'absence': absence_indices}
                    if len(presence_indices) < 1:
                        print(f"{taxon}, no presence!")
                    elif len(absence_indices) < 1:
                        print(f"{taxon}, no absence!")
                    else:
                        num_present = len(presence_indices)
                        count += 1
                        print(f"{taxon}, {count}, present cells: {num_present}")

                        init_pos_inds = random.sample(list(presence_indices), 1)
                        init_neg_inds = random.sample(list(absence_indices), 1)
                        sampled_idx = np.array([init_pos_inds, init_neg_inds])
                        init_pos_loc = np.float32(filtered_obs_locs[init_pos_inds])
                        init_neg_loc = np.float32(filtered_obs_locs[init_neg_inds])
                        locs = np.vstack((init_pos_loc, init_neg_loc))
                        idx = count - 1
                        pos_labs = np.array([1,1])
                        neg_labs = np.array([0,0])
                        labels = np.vstack((pos_labs,neg_labs))
                        data[int(taxon)] = {}
                        data[int(taxon)]['data'] = locs
                        data[int(taxon)]['class_idx'] = idx
                        data[int(taxon)]['sampled_idx'] = sampled_idx
                        data[int(taxon)]['labels'] = labels

            save_name = f"{save_loc}_{i+1}"
            np.savez(data=data, file=save_name)
        return

    def set_data_list_iteration(self, iteration):
        for taxon_id in self.data_lists.keys():
            # reset data to a specified iteration
            old_data = self.data_lists[taxon_id]['data']
            old_labs = self.data_lists[taxon_id]['labels']
            old_idx = self.data_lists[taxon_id]['sampled_idx']
            new_data = old_data[:iteration+2]
            new_labs = old_labs[:iteration+2]
            new_idx = old_idx[:iteration+2]
            self.data_lists[taxon_id]['data'] = new_data
            self.data_lists[taxon_id]['labels'] = new_labs
            self.data_lists[taxon_id]['sampled_idx'] = new_idx
        return
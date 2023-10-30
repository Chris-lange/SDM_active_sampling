from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import copy

class SamplingStrategy:
    def __init__(self, args, data_manager):
        self.args = args
        self.data_manager = data_manager
        self.sampling_method = self.args['sampling_method']
        self.batch_size = self.args['batch_size']

    def generate_taxa_sampling_mask(self, taxon_id, obs_locs):
        """
        Generate a class mask based on presence and absence information.
        Additionally, remove locations already visited.
        """
        # Presence and Absence Masking
        taxon_presence_mask = np.isin(range(len(obs_locs)), self.data_manager.gt_data['taxa_data'][taxon_id]['presence'])
        taxon_absence_mask = np.isin(range(len(obs_locs)), self.data_manager.gt_data['taxa_data'][taxon_id]['absence'])
        clss_mask = taxon_presence_mask | taxon_absence_mask

        if taxon_id in self.data_manager.data_lists:
            visited_indices = self.data_manager.data_lists[taxon_id]['sampled_idx']
            visited_indices = list(visited_indices.flatten())
            clss_mask[visited_indices] = False

        return clss_mask

    def sample(self, model=None, base_model=None, additional_classifiers=None):

        # Choose the appropriate sampling strategy
        if self.sampling_method == 'random_sampling':
            return self.random_sampling()

        elif self.sampling_method in ['most_positive_pred', 'most_uncertain_pred', 'largest_expected_model_change']:
            return self.selected_sampling(method=self.sampling_method, model=model)

        elif self.sampling_method == 'query_by_HS_committee':
            return self.query_by_HS_committee(base_model=base_model, additional_classifiers=additional_classifiers)

        elif self.sampling_method == 'query_by_committee':
            return self.query_by_committee(base_model=base_model)

        else:
            print('Sampling method not recognized!')
            raise NotImplementedError

    def random_sampling(self):
        sample_dict = {}

        for taxon_id, class_idx in self.data_manager.taxa_map.items():
            # generate valid sampling locs for this class
            clss_mask = self.generate_taxa_sampling_mask(taxon_id=taxon_id, obs_locs=self.data_manager.gt_data['locs'])
            valid_locs = self.data_manager.gt_data['locs'][clss_mask]
            filtered_to_original_idx_map = {filtered_idx:orig_idx for filtered_idx, orig_idx in
                                        enumerate(np.where(clss_mask)[0])}
            # Randomly sample a location for this taxon
            sample_dict[taxon_id] = {}
            inds = (np.random.choice(range(len(valid_locs)), 1, replace=False)).item()
            sample_dict[taxon_id]['location'] = valid_locs[inds]
            sample_dict[taxon_id]['idx'] = filtered_to_original_idx_map[inds]

        return sample_dict

    def selected_sampling(self, method, model):
        sample_dict = {}

        for taxon_id, class_idx in self.data_manager.taxa_map.items():
            # generate valid sampling locs for this class
            clss_mask=self.generate_taxa_sampling_mask(taxon_id=taxon_id, obs_locs=self.data_manager.gt_data['locs'])

            filtered_to_original_idx_map = {filtered_idx:orig_idx for filtered_idx, orig_idx in
                                        enumerate(np.where(clss_mask)[0])}

            # Extract Features
            valid_feats = self.data_manager.locs_to_feats[clss_mask]
            valid_feats = torch.from_numpy(valid_feats).to(self.args['device'])
            num_valid_locs = len(valid_feats)
            valid_preds = np.zeros(num_valid_locs)

            # Generate predictions in batches to avoid memory issues
            # only one taxon ata a time so increase batch size as operations are small
            for i in range(0, num_valid_locs, self.batch_size*10):
                end = min(i + self.batch_size*10, num_valid_locs)
                batch_feats = valid_feats[i:end]

                with torch.no_grad():
                    batch_preds = model(batch_feats, class_of_interest=self.data_manager.taxa_map[taxon_id], use_feats_as_input=True).cpu().numpy()

                valid_preds[i:end] = batch_preds

            # Sample Selection
            if method == 'most_positive_pred':
                index = valid_preds.argmax()
            elif method == 'most_uncertain_pred':
                index = (np.abs(valid_preds - 0.5)).argmin()
            elif method == 'largest_expected_model_change':
                model.to(self.args['device']).train()
                model.zero_grad()

                num_valid_locs = len(valid_feats)
                expected_change = torch.zeros(num_valid_locs).to(self.args['device'])

                for i in range(0, num_valid_locs, self.args['batch_size']):
                    end = min(i + self.args['batch_size'], num_valid_locs)
                    batch_feats = valid_feats[i:end]

                    logits = model(batch_feats, class_of_interest=self.data_manager.taxa_map[taxon_id],
                                   use_feats_as_input=True,
                                   return_logits=True)
                    logits = Variable(logits, requires_grad=True)

                    loss_pos = binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
                    loss_neg = binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')

                    loss_pos.backward(torch.ones_like(logits), retain_graph=True)
                    grad_pos = logits.grad.clone().detach()

                    logits.grad.zero_()

                    loss_neg.backward(torch.ones_like(logits), retain_graph=True)
                    grad_neg = logits.grad.clone().detach()

                    logits.grad.zero_()

                    prob_pos = torch.sigmoid(logits)
                    prob_neg = 1 - prob_pos

                    batch_expected_change = prob_pos * torch.abs(grad_pos) + prob_neg * torch.abs(grad_neg)
                    expected_change[i:end] = batch_expected_change

                index = int(expected_change.argmax())

            else:
                print('Sampling method not recognised!')
                raise NotImplementedError

            # Add sampled data to sample_dict
            sample_dict[taxon_id] = {}

            selected_sample_location = self.data_manager.gt_data['locs'][clss_mask][index].reshape(1, -1)
            selected_sample_idx = filtered_to_original_idx_map[index]
            sample_dict[taxon_id]["location"] = selected_sample_location
            sample_dict[taxon_id]["idx"] = selected_sample_idx
            print(f"location selected for class {class_idx}, taxon {taxon_id}")

        return sample_dict


    # Helper function to create a new model containing only the top classifiers
    def create_optimized_model(self, base_model, additional_classifiers, top_classifier_indices, device, taxon):
        # Create a new model (assumes that 'base_model' can be deep-copied)
        import copy
        optimized_model = copy.deepcopy(base_model)

        # Extract all the weights from the original models
        all_class_weights_base = base_model.class_emb.weight
        all_class_weights_additional = None
        if additional_classifiers is not None:
            all_class_weights_additional = additional_classifiers.class_emb.weight[self.data_manager.taxa_map[taxon], :]

        # Combine base and additional classifiers' weights
        if additional_classifiers is not None:
            all_class_weights = torch.cat((all_class_weights_base, all_class_weights_additional.reshape(1, -1)), dim=0)
        else:
            all_class_weights = all_class_weights_base

        # Select only the top classifiers based on their indices
        top_class_weights = all_class_weights[top_classifier_indices, :]

        # Create the new class_emb layer
        class_emb = torch.nn.Linear(top_class_weights.shape[1], top_class_weights.shape[0], bias=False)

        # Initialize the new layer's weights and biases (if applicable)
        class_emb.weight.data.zero_()

        # Replace the old class_emb layer with the new one
        optimized_model.class_emb = class_emb

        # Update the weights in the optimized model
        optimized_model.class_emb.weight.data = top_class_weights.to(device)

        return optimized_model

    # Helper function to evaluate the model in batches
    def eval_model_in_batches(self, model, features, batch_size, taxon_of_interest=None):
        num_data = len(features)
        preds = []
        for i in range(0, num_data, batch_size):
            end = min(i + batch_size, num_data)
            batch_feats = features[i:end]
            with torch.no_grad():
                if taxon_of_interest == None:
                    batch_preds = model(batch_feats, use_feats_as_input=True)
                else:
                    batch_preds = model(batch_feats, use_feats_as_input=True,
                                        class_of_interest=self.data_manager.taxa_map[taxon_of_interest])
                preds.append(batch_preds.cpu())
        return torch.cat(preds).numpy()


    # Main function
    def query_by_HS_committee(self, base_model, additional_classifiers=None):
        print("sampling using weighted hypothesis set committee")
        max_committee_members = self.args['max_committee_size']
        sample_dict = {}
        base_model.eval()
        if additional_classifiers:
            additional_classifiers.eval()

        for taxon, clss in self.data_manager.taxa_map.items():
            class_mask = self.generate_taxa_sampling_mask(taxon, self.data_manager.gt_data['locs'])
            sampled_feats, _, pos_length = self.data_manager.get_sampled_feats_and_labels(taxon_id=taxon)
            sampled_feats = torch.from_numpy(sampled_feats).to(self.args['device'])

            filtered_to_original_idx_map = {filtered_idx:orig_idx for filtered_idx, orig_idx in
                                        enumerate(np.where(class_mask)[0])}

            # Evaluate base_model and additional_classifiers in batches
            base_model_probs = self.eval_model_in_batches(base_model, sampled_feats, self.batch_size)
            if additional_classifiers:
                additional_classifier_probs = self.eval_model_in_batches(additional_classifiers, sampled_feats,
                                                                    self.batch_size, taxon)
                sampled_probs = np.hstack((base_model_probs, additional_classifier_probs.reshape(-1, 1)))
            else:
                sampled_probs = base_model_probs

            # A good committee member will have (1-prob) = 1 (approx) for "absent" labeled datapoints
            combined_probs = np.concatenate([sampled_probs[:pos_length, :], 1 - sampled_probs[pos_length:, :]])

            # Compute classifier weights
            probability_method = self.args.get('probability_combination_method', 'product')
            if probability_method == 'product':
                classifier_weights = np.prod(combined_probs, axis=0)
            elif probability_method == 'average':
                classifier_weights = np.mean(combined_probs, axis=0)
            else:
                raise ValueError("Probability combination method not recognised")

            if max_committee_members != -1:
                top_classifier_indices = np.argsort(classifier_weights)[-max_committee_members:]
                committee_weights = np.sort(classifier_weights)[-max_committee_members:]
            else:
                top_classifier_indices = np.argsort(classifier_weights)
                committee_weights = np.sort(classifier_weights)

            # Create a new model containing only the top classifiers
            optimized_model = self.create_optimized_model(base_model=base_model,
                                                          additional_classifiers=additional_classifiers,
                                                          top_classifier_indices=top_classifier_indices,
                                                          device=self.args['device'], taxon=taxon)

            # Move optimized_model to the device
            optimized_model.to(self.args['device']).eval()

            valid_feats = torch.from_numpy(self.data_manager.locs_to_feats[class_mask]).to(self.args['device'])
            num_valid_locs = len(valid_feats)
            sums = np.zeros(num_valid_locs)

            for i in range(0, num_valid_locs, self.batch_size):
                end = min(i + self.batch_size, num_valid_locs)
                batch_feats = valid_feats[i:end]

                # Evaluate optimized_model in batches
                batch_preds = self.eval_model_in_batches(optimized_model, batch_feats, self.batch_size)

                # for soft voting
                if self.args['vote_type'] == 'soft':
                    soft_batch_preds = (batch_preds - 0.5) * 2

                    # if member agreement with data is considered
                    if self.args['member_weighting'] == 'agreement':
                        soft_batch_preds *= committee_weights

                    # otherwise do not consider agreement
                    elif self.args['member_weighting'] == 'equal':
                        pass
                    else:
                        print('Member weighting not recognised')
                        raise NotImplementedError
                    sums[i:end] += np.sum(soft_batch_preds, axis=1)

                # for hard voting
                elif self.args['vote_type'] == 'hard':
                    binary_preds = np.where(batch_preds > 0.5, 1, -1)

                    # if member agreement with data is considered
                    if self.args['member_weighting'] == 'agreement':
                        binary_preds = binary_preds.astype(np.float32)
                        binary_preds *= committee_weights

                    # otherwise do not consider agreement
                    elif self.args['member_weighting'] == 'equal':
                        pass
                    else:
                        print('Member weighting not recognised')
                        raise NotImplementedError
                    sums[i:end] += np.sum(binary_preds, axis=1)
                else:
                    print('Voting type not recognised')
                    raise NotImplementedError

            ind = np.argmin(np.abs(sums))
            sample_loc = np.reshape(self.data_manager.gt_data['locs'][class_mask][ind], (1, -1))
            sample_dict[taxon] = {}
            sample_dict[taxon]["location"] = sample_loc
            sample_dict[taxon]["idx"] = filtered_to_original_idx_map[ind]
            print(f"location selected for class {self.data_manager.taxa_map[taxon]} taxon {taxon}")

        return sample_dict


    def split_data_for_taxon(self, sampled_feats, labels):
        subsets = []
        max_members = self.args['max_committee_size']
        n = len(sampled_feats)

        # Helper function to check if both classes are present
        def check_classes(subset_labels):
            return len(set(subset_labels)) > 1

        for i in range(n):
            # Create new subset by excluding i-th point
            subset_feats = np.delete(sampled_feats, i, axis=0)
            subset_labels = np.delete(labels, i, axis=0)

            # Check if both classes are present
            if check_classes(subset_labels):
                subsets.append((subset_feats, subset_labels))

        # Limit the number of subsets to max_members if necessary
        if len(subsets) > max_members:
            subsets = subsets[:max_members]

        # Always include the whole dataset as one of the subsets
        subsets.append((sampled_feats, labels))

        return subsets

    def train_committee_members(self, subsets, base_model):
        # Step 1: Initialize with a deep copy of the base model
        optimized_model = copy.deepcopy(base_model)

        # Step 2: Create a new class_emb layer
        input_dim = base_model.class_emb.weight.shape[1]
        output_dim = len(subsets)  # Number of committee members
        new_class_emb = torch.nn.Linear(input_dim, output_dim, bias=False)
        new_class_emb.weight.data.zero_()  # Initialize with zeros

        # Step 3: Train and update
        for i, (subset_feats, subset_labels) in enumerate(subsets):
            if len(np.unique(subset_labels)) >= 2:
                clf = LogisticRegression(random_state=self.args['seed'], fit_intercept=False, max_iter=250)
                clf.fit(subset_feats, subset_labels)

                new_weights = torch.from_numpy(clf.coef_[0, :].astype(np.float32)).to(device=self.args['device'])
                new_class_emb.weight.data[i, :] = new_weights

        # Step 4: Replace the old class_emb layer with the new one
        optimized_model.class_emb = new_class_emb

        return optimized_model

    def query_by_committee(self, base_model):
        print("Sampling using traditional Query by Committee")
        sample_dict = {}
        for taxon, _ in self.data_manager.taxa_map.items():
            class_mask = self.generate_taxa_sampling_mask(taxon, self.data_manager.gt_data['locs'])
            sampled_feats, labels, pos_length = self.data_manager.get_sampled_feats_and_labels(taxon_id=taxon)
            sampled_feats = torch.from_numpy(sampled_feats).to(self.args['device'])
            subsets = self.split_data_for_taxon(sampled_feats, labels)
            committee_members = self.train_committee_members(subsets, base_model)

            filtered_to_original_idx_map = {filtered_idx: orig_idx for filtered_idx, orig_idx in
                                            enumerate(np.where(class_mask)[0])}

            # Evaluate base_model and additional_classifiers in batches

            sampled_probs = self.eval_model_in_batches(committee_members, sampled_feats,
                                                                     self.batch_size)

            # Check if sampled_probs is 1-dimensional
            if len(sampled_probs.shape) == 1:
                # Reshape it to 2D
                sampled_probs = sampled_probs.reshape(-1, 1)

            # A good committee member will have (1-prob) = 1 (approx) for "absent" labeled datapoints
            combined_probs = np.concatenate([sampled_probs[:pos_length, :], 1 - sampled_probs[pos_length:, :]])

            # Compute classifier weights
            probability_method = self.args.get('probability_combination_method', 'product')
            if probability_method == 'product':
                classifier_weights = np.prod(combined_probs, axis=0)
            elif probability_method == 'average':
                classifier_weights = np.mean(combined_probs, axis=0)
            else:
                raise ValueError("Probability combination method not recognised")


            top_classifier_indices = np.argsort(classifier_weights)
            committee_weights = np.sort(classifier_weights)

            valid_feats = torch.from_numpy(self.data_manager.locs_to_feats[class_mask]).to(self.args['device'])
            num_valid_locs = len(valid_feats)
            sums = np.zeros(num_valid_locs)

            for i in range(0, num_valid_locs, self.batch_size):
                end = min(i + self.batch_size, num_valid_locs)
                batch_feats = valid_feats[i:end]

                # Evaluate optimized_model in batches
                batch_preds = self.eval_model_in_batches(committee_members, batch_feats, self.batch_size)

                # for soft voting
                if self.args['vote_type'] == 'soft':
                    soft_batch_preds = (batch_preds - 0.5) * 2

                    # if member agreement with data is considered
                    if self.args['member_weighting'] == 'agreement':
                        soft_batch_preds *= committee_weights

                    # otherwise do not consider agreement
                    elif self.args['member_weighting'] == 'equal':
                        pass
                    else:
                        print('Member weighting not recognised')
                        raise NotImplementedError
                    sums[i:end] += np.sum(soft_batch_preds, axis=1)

                # for hard voting
                elif self.args['vote_type'] == 'hard':
                    binary_preds = np.where(batch_preds > 0.5, 1, -1)

                    # if member agreement with data is considered
                    if self.args['member_weighting'] == 'agreement':
                        binary_preds *= committee_weights

                    # otherwise do not consider agreement
                    elif self.args['member_weighting'] == 'equal':
                        pass
                    else:
                        print('Member weighting not recognised')
                        raise NotImplementedError
                    sums[i:end] += np.sum(binary_preds, axis=1)
                else:
                    print('Voting type not recognised')
                    raise NotImplementedError

            ind = np.argmin(np.abs(sums))
            sample_loc = np.reshape(self.data_manager.gt_data['locs'][class_mask][ind], (1, -1))
            sample_dict[taxon] = {}
            sample_dict[taxon]["location"] = sample_loc
            sample_dict[taxon]["idx"] = filtered_to_original_idx_map[ind]
            print(f"location selected for class {self.data_manager.taxa_map[taxon]} taxon {taxon}, committee_members = {len(top_classifier_indices)}")

        return sample_dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

class UpdateStrategy:
    def __init__(self, args, data_manager):
        self.args = args
        self.data_manager = data_manager
        self.update_method = self.args['update_method']

    def update(self, model=None, base_model=None, additional_classifiers=None):

        # Choose the appropriate update strategy
        if self.update_method == 'logistic_regression':
            return self.logistic_regression(model=model), additional_classifiers

        elif self.update_method in ['weighted_average', 'weighted_average_plus']:
            return self.weighted_average_update(base_model=base_model,
                                                active_sampling_model=model,
                                                additional_classifiers=additional_classifiers)
        else:
            print('Update method not recognized!')
            raise NotImplementedError

    def logistic_regression(self, model):
        for taxon_id, class_idx in self.data_manager.taxa_map.items():
            # get feats and labels for sampled locations
            sampled_feats, labels, pos_length = self.data_manager.get_sampled_feats_and_labels(taxon_id=taxon_id)

            if len(np.unique(labels)) >= 2:
                clf = LogisticRegression(random_state=self.args['seed'], fit_intercept=False,
                                         max_iter=250).fit(sampled_feats, labels)

                new_weights = torch.from_numpy(clf.coef_[0, :].astype(np.float32)).to(device=self.args['device'])
                model.class_emb.weight.data[class_idx, :] = new_weights
            else:
                print(f"Only one class found for class {class_idx}, taxon {taxon_id}. Skipping update.")
        return model

    def update_additional_classifiers(self, additional_classifiers):
        print('updating additional classifiers using logistic regression method')
        additional_classifiers = self.logistic_regression(additional_classifiers)
        return additional_classifiers

    def weighted_average_update(self, base_model, active_sampling_model, additional_classifiers):
        base_model.eval()

        weights_array = np.zeros((500,44181))

        # Perform logistic regression update on additional classifiers if they exist (I.e. for WA_HSS+)
        if additional_classifiers is not None:
            additional_classifiers = self.update_additional_classifiers(additional_classifiers)
            additional_classifiers.eval()

        # Iterate through each taxon
        for taxon_id, class_idx in self.data_manager.taxa_map.items():

            # Get the sampled features and labels for this taxon
            sampled_feats, labels, pos_length = self.data_manager.get_sampled_feats_and_labels(taxon_id=taxon_id)
            sampled_feats = torch.from_numpy(sampled_feats).to(self.args['device'])

            # get probabilities for all classifiers
            # Don't need to batch as number of samples is small
            with torch.no_grad():
                base_model_probs = base_model(sampled_feats, use_feats_as_input=True)
            base_model_probs = np.array(base_model_probs.cpu())

            # If we have additional classifiers we should also evaluate these
            if additional_classifiers is not None:
                with torch.no_grad():
                    additional_classifier_probs = additional_classifiers(sampled_feats, use_feats_as_input=True,
                                                                         class_of_interest=self.data_manager.taxa_map[taxon_id])
                additional_classifier_probs = np.array(additional_classifier_probs.cpu())

                # combine base model and additional classifiers
                sampled_probs = np.hstack((base_model_probs, additional_classifier_probs.reshape(-1, 1)))
            else:
                sampled_probs = base_model_probs

            # Invert probabilities for negative examples
            sampled_probs[pos_length:] = 1 - sampled_probs[pos_length:]

            # Combine probabilities based on method
            prob_method = self.args.get('probability_combination_method', 'product')
            if prob_method == 'product':
                combined_probs = np.prod(sampled_probs, axis=0)
            elif prob_method == 'average':
                combined_probs = np.mean(sampled_probs, axis=0)
            else:
                print('Probability combination method not recognised')
                raise NotImplementedError

            # consider all classifiers
            if additional_classifiers is not None:
                all_class_weights = np.vstack((np.array(base_model.class_emb.weight.cpu().detach()),
                                             np.array(additional_classifiers.class_emb.weight[class_idx, :].cpu().detach())))
            else:
                all_class_weights = np.array(base_model.class_emb.weight.cpu().detach())

            # Combine classifiers according to their weights
            weights_array[class_idx,:] = combined_probs
            new_class_weight = np.average(all_class_weights, axis=0, weights=combined_probs)
            active_sampling_model.class_emb.weight.data[class_idx, :] = torch.from_numpy(new_class_weight).to(self.args['device'])

        return active_sampling_model, additional_classifiers
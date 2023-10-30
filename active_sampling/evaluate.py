import numpy as np
import torch
from sklearn.metrics import average_precision_score

class ModelEvaluator:
    def __init__(self, args, data_manager, initial_results=None):
        self.args = args
        self.results = initial_results if initial_results else {}
        self.data_manager = data_manager
        self.batch_size = self.args['batch_size']


    def eval(self, model, update_steps):
        print('Evaluating...')
        # Check if results for this update step already exist
        if update_steps in self.results:
            print(f"Results for update step {update_steps} already exist. Skipping evaluation.")
            return
        results = self.eval_sampled_taxa(model=model)
        self.results[update_steps] = results  # Store results keyed by update steps
        return

    def eval_sampled_taxa(self, model):
        print('Evaluating model on ground truth')

        results = {'mean_average_precision': {},
                   'num_eval_species_total': len(self.data_manager.taxa_map)}

        for taxon, class_index in self.data_manager.taxa_map.items():
            # Extract presence and absence indices from the ground truth data
            presence_indices = self.data_manager.gt_data['taxa_data'][taxon]['presence']
            absence_indices = self.data_manager.gt_data['taxa_data'][taxon]['absence']

            # Create combined mask of presence and absence indices
            combined_mask = np.concatenate((presence_indices, absence_indices))

            # Create a mapping from original indices to new indices
            original_to_new_idx_map = {original_idx: new_idx for new_idx, original_idx in enumerate(combined_mask)}

            # Extract features using the combined mask
            # valid_feats = np.array([self.data_manager.locs_to_feats[tuple(loc)] for loc in self.data_manager.gt_data['locs'][combined_mask]])
            valid_feats = self.data_manager.locs_to_feats[combined_mask]
            valid_feats = torch.from_numpy(valid_feats).to(self.args['device'])

            model.to(self.args['device']).eval()

            num_valid_locs = len(valid_feats)
            valid_preds = np.zeros((num_valid_locs))

            # Generate predictions in batches to avoid memory issues
            # Use much larger batch as this operation only involves a single taxon - matrices are small
            for i in range(0, num_valid_locs, self.batch_size*10):
                end = min(i + self.batch_size*10, num_valid_locs)
                batch_feats = valid_feats[i:end]

                with torch.no_grad():
                    batch_preds = model(batch_feats, class_of_interest=self.data_manager.taxa_map[taxon], use_feats_as_input=True).cpu().numpy()

                valid_preds[i:end] = batch_preds

            # Prepare ground truth labels
            gt = np.zeros(len(combined_mask), dtype=np.float32)
            presence_new_indices = [original_to_new_idx_map[idx] for idx in presence_indices]
            gt[presence_new_indices] = 1.0

            # Compute the AP for this taxon
            ap = average_precision_score(gt, valid_preds)
            results['mean_average_precision'][taxon] = ap  # Store AP by taxon ID

            print(f'AP for class {class_index}, taxon {taxon}: {np.round(ap, 3)}')

        results['per_species_average_precision'] = results['mean_average_precision']
        results['mean_average_precision'] = np.mean(list(results['mean_average_precision'].values()))

        print(f'Evaluation finished')
        print(f'Mean average precision: {results["mean_average_precision"]}')

        return results


    def get_class(self, taxa, class_to_taxa):
        clss = None
        for i, id in enumerate(class_to_taxa):
            if id == taxa:
                clss = i
        if clss == None:
            print("clss not found")
            raise Exception
        return clss

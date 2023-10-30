import numpy as np
import json

# One time use function to update old init data to new "data_list" format
def convert_and_save_data_to_data_list(old_data_loc, new_data_loc, taxa_list):
    # Load old data
    old_data = np.load(old_data_loc, allow_pickle=True)['data'].item()

    # Initialize empty data_list dictionary
    data_list = {}

    for idx, (key, value) in enumerate(old_data.items()):
        taxon_id = taxa_list[idx]  # Get the actual taxon_id from taxa_list
        pos, neg = value
        pos_labs = np.ones((pos.shape[0], 2), dtype=int)
        neg_labs = np.zeros((neg.shape[0], 2), dtype=int)
        combined_data = np.vstack((pos, neg))
        combined_labs = np.vstack((pos_labs, neg_labs))
        data_list[taxon_id] = {'idx': idx, 'data': combined_data, 'labels': combined_labs}

    # Save data_list
    np.savez(new_data_loc, data=data_list)

def load_taxa_list(taxa_list_loc):
    return list(np.load(taxa_list_loc, allow_pickle=True))

def save_taxa_list(taxa_list, save_loc):
    np.save(file=save_loc, arr=np.array(taxa_list))

# Create config JSON for arguments that are unlikely to change
def create_config(save_loc):
    config = {
        'model_path': '../backbone/models/no_cap_coord_active_sample_backbone.pt',
        'use_linear_model': 0,
        'batch_size': 1024,
        'env_loc': '',
        'save_dir': '../results',
        'ocean_mask_loc': '../data/ocean_mask_hr.npy',
        'init_data_loc': {
            'IUCN': ['../data/init_data/iucn_init_1.npz',
                     '../data/init_data/iucn_init_2.npz',
                     '../data/init_data/iucn_init_3.npz'],
            'SNT': ['../data/init_data/snt_init_1.npz',
                    '../data/init_data/snt_init_2.npz',
                    '../data/init_data/snt_init_3.npz']
        },
        'taxa_list_loc': {
            'IUCN': '../data/500_IUCN_taxa.npy',
            'SNT': '../data/500_SNT_taxa.npy'
        },
        'gt_data': {
            'IUCN': '../data_from_sinr_repo/eval/iucn/iucn_res_5.json',
            'SNT': '../data_from_sinr_repo/eval/snt/snt_res_5.npy'
        },
        'eval_type': {
            'IUCN': 'IUCN',
            'SNT': 'SNT'
        },
        'seeds': [999, 1000, 1001]
    }
    # Write the configuration to a JSON file
    with open(save_loc, 'w') as file:
        json.dump(config, file, indent=4)
    return


# def mask_to_locations(mask, lat_range=(-1, 1), lon_range=(-1, 1)):
#
#     # Identify the indices of land pixels
#     land_indices = np.argwhere(mask == 1)
#
#     # Get mask shape
#     rows, cols = mask.shape
#
#     # Scaling factors for converting index to [-1, 1]
#     lat_scale = (lat_range[1] - lat_range[0]) / (rows - 1)
#     lon_scale = (lon_range[1] - lon_range[0]) / (cols - 1)
#
#     # Convert indices to lat, lon within the range [-1, 1]
#     loc_to_mask_dict = {}
#     locations = []
#     for i, j in land_indices:
#         lat = i * lat_scale + lat_range[0]
#         lon = j * lon_scale + lon_range[0]
#         locations.append((lon, lat))
#         loc_to_mask_dict[(np.float32(lon),np.float32(lat))] = j,i
#
#     return locations, loc_to_mask_dict
#

def mask_to_locations(mask, lat_range=(-90, 90), lon_range=(-180, 180)):
    # Identify the indices of land pixels
    land_indices = np.argwhere(mask == 1)

    # Get mask shape
    rows, cols = mask.shape

    # Scaling factors for converting index to [-90, 90] for lat and [-180, 180] for lon
    lat_scale = (lat_range[1] - lat_range[0]) / (rows - 1)
    lon_scale = (lon_range[1] - lon_range[0]) / (cols - 1)

    # Convert indices to lat, lon within the range [-90, 90] for lat and [-180, 180] for lon
    loc_to_mask_dict = {}
    locations = []
    for i, j in land_indices:
        lat = (rows - 1 - i) * lat_scale + lat_range[0]  # Flip latitude indexing
        lon = j * lon_scale + lon_range[0]
        locations.append((lon, lat))
        loc_to_mask_dict[(np.float32(lon), np.float32(lat))] = j, i

    return locations, loc_to_mask_dict

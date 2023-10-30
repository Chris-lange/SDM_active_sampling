import json
import numpy as np
import h3
import time

t = time.time()

with open('snt2hex_res_5.json', 'r') as f:
    data = json.load(f)

gbif_ids = []
obs_locs = []
obs_locs_idx = []
for loc_idx, hx in enumerate(data): # Note: This iteration ordering is guaranteed in Python 3.7+
    for sp in data[hx]:
        gbif_ids.append(sp)
    cur_lat, cur_lon = h3.h3_to_geo(hx)
    obs_locs.append([cur_lon, cur_lat])
    obs_locs_idx.append(loc_idx)
obs_locs = np.array(obs_locs).astype(np.float32)
taxa = np.sort(np.unique(gbif_ids).astype(np.int32)) # official ordering for eval purposes
loc_indices_per_species = [[] for i in range(len(taxa))] # Locations where the species is known to be present or absent.
labels_per_species = [[] for i in range(len(taxa))] # Present or absent determination for each location.
for loc_idx, hx in enumerate(data):
    for sp in data[hx]:
        sp_idx = np.where(taxa == int(sp))[0][0]
        loc_indices_per_species[sp_idx].append(loc_idx)
        cur_label = int(len(data[hx][sp]) > 0)
        labels_per_species[sp_idx].append(cur_label)

out = {
    'loc_indices_per_species': loc_indices_per_species,
    'labels_per_species': labels_per_species,
    'taxa': taxa,
    'obs_locs': obs_locs,
    'obs_locs_idx': obs_locs_idx
}

print(f'data prep completed in {np.around((time.time() - t)/60.0)} min.')

t = time.time()
np.save('all_sps_flat_time_res5_fmt.npy', out)
print(f'data save completed in {np.around((time.time() - t)/60.0)} min.')
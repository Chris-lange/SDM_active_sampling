# Instructions for Data Preparation
After following these instructions, the `data` directory should have the following structure:
```
data
├── README.md
├── environmental_variables
│   ├── bioclim_elevation_scaled.npy
│   └── format_env_feats.py
├── iucn
│   ├── iucn_res_5.json
└── snt
│   ├── transform_snt_data.py
│   └── snt_res_5.npy
├── init_data
│   ├── iucn_init_1.npz
│   ├── iucn_init_2.npz
│   ├── iucn_init_3.npz
│   ├── snt_init_1.npz
│   ├── snt_init_2.npz
│   └── snt_init_3.npz
└── masks
│   ├── ocean_mask.npy
│   └── ocean_mask_hr.npy
└── taxa_lists
    ├── 500_IUCN_taxa.npy
    └── 500_SNT_taxa.npy
```

## Training & Evaluation Data

1. Navigate to the `SDM_active_sampling/data` directory:

```bash
cd /path/to/SDM_active_sampling/data/
```

2. Download the data file:
```bash
curl -L https://data.caltech.edu/records/b0wyb-tat89/files/data.zip --output data.zip
```

3. Extract the data:
```bash
unzip -q data.zip
```

4. Move data into specified directories:

```bash
mkdir iucn
mkdir snt
mv data/eval/iucn/iucn_res_5.json iucn/
mv data/eval/snt/snt_res_5.npy snt/
```

5. Clean up:
```bash
rm data.zip
rm -rf data/
```

## Environmental Features

1. Navigate to the directory for the environmental features:
```bash
cd /path/to/SDM_active_sampling/data/environmental_variables
```

2. Download the data:
```bash
curl -L https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_5m_bio.zip --output wc2.1_5m_bio.zip
curl -L https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_5m_elev.zip --output wc2.1_5m_elev.zip
```

3. Extract the data:
```bash
unzip -q wc2.1_5m_bio.zip
unzip -q wc2.1_5m_elev.zip
```

4. Activate the `active_learning` environment
```bash
 conda activate active_learning
```

4. Run the formatting script:
```bash
python format_env_feats.py
```

5. Clean up:
```bash
rm *.zip
rm *.tif
```

## Pretrained Backbone Models



You may download pretrained models to recreate the results in the paper from here:

https://uoe-my.sharepoint.com/:f:/g/personal/s2125675_ed_ac_uk/EjZbGUCc1gZBo2x0T-PQz4wBtsGhrl82Y7MMa-q3OeW0JA?e=pyBaUI

After downloading, unzip and move the necessary models into the `models` directory within `backbone`.

After following these instructions, the `backbone` directory should have the following structure: 

```
backbone
├── backbone_utils.py
├── models.py
└── models
   ├── pretrained_model_1
   ├── pretrained_model_2
   ├── pretrained_model_3
   └── ...


```

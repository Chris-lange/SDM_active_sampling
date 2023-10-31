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

1. Download the data file:
```bash
curl -L https://data.caltech.edu/records/b0wyb-tat89/files/data.zip --output data.zip
```

2. Extract the data and clean up:
```bash
unzip -q data.zip
```

3. Clean up:
```bash
rm data.zip
```

4. Delete unnecessary files and move data into directories above to work with current `config.json` or modify `config.json` to point to current directories.

## Environmental Features

1. Navigate to the directory for the environmental features:
```
cd /path/to/sinr/data/env
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

4. Run the formatting script:
```bash
python format_env_feats.py
```

5. Clean up:
```bash
rm *.zip
rm *.tif
```
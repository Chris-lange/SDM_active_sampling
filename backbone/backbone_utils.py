import numpy as np
import torch
import torch.nn as nn
import math
from . import models


def bilinear_interpolate(loc_ip, data, remove_nans=False, device='cpu'):
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    # map to [0,1], then scale to data size
    loc = (loc_ip.clone() + 1) / 2.0
    loc[:,1] = 1 - loc[:,1] # this is because latitude goes from +90 on top to bottom while
                            # longitude goes from -90 to 90 left to right
    if remove_nans:
        loc[torch.isnan(loc)] = 0.0
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long().to(device)  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    data=data.to(device)
    dx=dx.to(device)
    dy=dy.to(device)

    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val


def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T


def load_context_feats(data_path, device, normalize=False):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats[np.isnan(context_feats)] = 0  # NaNs
    context_feats = torch.from_numpy(context_feats).to(device)  # this could be a problem if very large

    if normalize:
        mu = context_feats.reshape((context_feats.shape[0]*context_feats.shape[1], context_feats.shape[2])).mean(0)
        std = context_feats.reshape((context_feats.shape[0]*context_feats.shape[1], context_feats.shape[2])).std(0)
        context_feats -= mu.unsqueeze(0).unsqueeze(0)
        context_feats /= std.unsqueeze(0).unsqueeze(0)

    return context_feats


def fourier_encoding(ip_feats, num_basis_feats, concat_dim):
    # see https://arxiv.org/pdf/2003.08934.pdf eqn (4)
    if ip_feats.shape[-1] == 1:
        scale = torch.arange(num_basis_feats, dtype=ip_feats.dtype, device=ip_feats.device)
    elif ip_feats.shape[-1] == 2:
        scale = torch.zeros((num_basis_feats*2), dtype=ip_feats.dtype, device=ip_feats.device)
        scale[0::2] = torch.arange(num_basis_feats)
        scale[1::2] = torch.arange(num_basis_feats)

    if len(ip_feats.shape) == 2:
        repeats = [1, num_basis_feats]
    elif len(ip_feats.shape) == 3:
        repeats = [1, 1, num_basis_feats]

    ip_feats_r = ip_feats.repeat(repeats)*math.pi*2**scale
    ip_feats_r = torch.cat((torch.sin(ip_feats_r), torch.cos(ip_feats_r)), concat_dim)
    return ip_feats_r


def encode_loc_date(loc_ip, date_ip, concat_dim=1, params=None, device='cpu'):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat
    if params['loc_encode'] == 'encode_cos_sin_basis':
        feats = fourier_encoding(loc_ip, params['num_cos_sin_basis'], concat_dim)

    if params['loc_encode'] == 'encode_none':
        feats = loc_ip

    elif params['loc_encode'] == 'encode_cos_sin':
        feats = torch.cat((torch.sin(math.pi*loc_ip), torch.cos(math.pi*loc_ip)), concat_dim)

    elif params['loc_encode'] == 'encode_cos_sin_basis':
        feats = fourier_encoding(loc_ip, params['num_cos_sin_basis'], concat_dim)

    elif params['loc_encode'] == 'encode_3D':
        # X, Y, Z in 3D space
        # notation selects the last column
        cos_lon = torch.cos(math.pi*loc_ip[..., 0]).unsqueeze(-1)
        sin_lon = torch.sin(math.pi*loc_ip[..., 0]).unsqueeze(-1)
        cos_lat = torch.cos(math.pi*loc_ip[..., 1]).unsqueeze(-1)
        sin_lat = torch.sin(math.pi*loc_ip[..., 1]).unsqueeze(-1)
        feats   = torch.cat((cos_lon*cos_lat, sin_lon*cos_lat, sin_lat), concat_dim)

    else:
        print('error - no loc feat type defined')


    if params['use_date_feats']:
        if date_ip.shape[-1] != 1:
            feats_date = date_ip.unsqueeze(-1)
        else:
            feats_date = date_ip

        if params['date_encode'] == 'encode_none':
            pass

        elif params['date_encode'] == 'encode_cos_sin':
            feats_date = torch.cat((torch.sin(math.pi*feats_date),
                                    torch.cos(math.pi*feats_date)), concat_dim)

        elif params['date_encode'] == 'encode_cos_sin_basis':
            feats_date = fourier_encoding(feats_date, params['num_cos_sin_basis'], concat_dim)

        else:
            print('error - no date feat type defined')

        feats = torch.cat((feats, feats_date), concat_dim).to(device)

    return feats


def convert_loc_and_date(locs, dates, device):
    # locs is in lon {-180, 180}, lat {90, -90}
    # output is in the range [-1, 1]

    # dates is in range [0, 1]
    # output is in the range [-1, 1]

    x_locs = locs.astype(np.float32)
    x_locs[:,0] /= 180.0
    x_locs[:,1] /= 90.0
    x_locs = torch.from_numpy(x_locs)
    x_locs = x_locs.to(device)

    x_dates = torch.from_numpy(dates.astype(np.float32)*2 - 1).to(device)

    return x_locs, x_dates


def generate_input_feats(locs, dates, params, context_feats=None, device='cpu'):
    # Expects locs and dates to be in [-1, 1]
    # Typically run after convert_loc_and_date()

    feats = encode_loc_date(locs, dates, params=params, device=device)

    if params['use_context_feats'] == 1 and context_feats is not None:
        context_feats = bilinear_interpolate(locs, context_feats, device=device)
        feats = torch.cat((feats, context_feats), 1).to(device)

    return feats


def generate_feats_array(model, params, args, context_feats, eval_data):
    print('Generating features from base model for each location')
    obs_locs = np.array(eval_data['locs'], dtype=np.float32)
    scaled_locs = obs_locs / np.array([180.0, 90.0], dtype=np.float32)
    batch_size = args['batch_size']

    loc_to_feats_array = np.zeros(shape=(len(obs_locs), params['num_filts']))

    model.to(args['device']).eval()

    for i in range(0, len(obs_locs), batch_size):
        end = min(i + batch_size, len(obs_locs))
        batch_scaled_locs = scaled_locs[i:end]

        backbone_input_feats = generate_input_feats(
            torch.from_numpy(batch_scaled_locs), [0], params, context_feats
        ).to(args['device'])

        with torch.no_grad():
            batch_backbone_output_feats = model(backbone_input_feats, return_feats=True).cpu().numpy()

        loc_to_feats_array[i:end] = batch_backbone_output_feats
    loc_to_feats_array = np.array(loc_to_feats_array, dtype=np.float32)

    return loc_to_feats_array


def generate_feats_for_mapping(model, params, args, locs, context_feats):
    print('Generating features from base model for each location')
    obs_locs = np.array(locs, dtype=np.float32)
    scaled_locs = obs_locs / np.array([180.0, 90.0], dtype=np.float32)
    batch_size = args['batch_size']

    loc_to_feats_array = np.zeros(shape=(len(obs_locs), params['num_filts']))

    model.to(args['device']).eval()

    for i in range(0, len(obs_locs), batch_size):
        end = min(i + batch_size, len(obs_locs))
        batch_scaled_locs = scaled_locs[i:end]

        backbone_input_feats = generate_input_feats(
            torch.from_numpy(batch_scaled_locs), [0], params, context_feats
        ).to(args['device'])

        with torch.no_grad():
            batch_backbone_output_feats = model(backbone_input_feats, return_feats=True).cpu().numpy()

        loc_to_feats_array[i:end] = batch_backbone_output_feats
    loc_to_feats_array = np.array(loc_to_feats_array, dtype=np.float32)

    return loc_to_feats_array


def generate_feats(model, params, args, context_feats, eval_data):
    print('Generating features from base model for each location')
    obs_locs = np.array(eval_data['locs'], dtype=np.float32)
    scaled_locs = obs_locs / np.array([180.0, 90.0], dtype=np.float32)
    batch_size = args['batch_size']

    loc_to_feats_dict = {}

    model.to(args['device']).eval()

    for i in range(0, len(obs_locs), batch_size):
        end = min(i + batch_size, len(obs_locs))
        batch_scaled_locs = scaled_locs[i:end]

        backbone_input_feats = generate_input_feats(
            torch.from_numpy(batch_scaled_locs), [0], params, context_feats
        ).to(args['device'])

        with torch.no_grad():
            batch_backbone_output_feats = model(backbone_input_feats, return_feats=True).cpu().numpy()

        batch_loc_to_feats = {tuple(loc): feat for loc, feat in zip(obs_locs[i:end], batch_backbone_output_feats)}
        loc_to_feats_dict.update(batch_loc_to_feats)

    return loc_to_feats_dict

def modify_model(model, params, taxa_map):
    """
    Modify the class layer of the backbone model to match the number of taxa being actively sampled.

    Parameters:
    - model: The original backbone model.
    - params: A dictionary of parameters related to the model.
    - taxa_map: A mapping from taxon IDs to class indices.

    Returns:
    - The modified model.
    - The updated parameters.
    """

    print('Replacing class layer of backbone model.')

    # Retrieve the number of input filters to the class_emb layer
    num_filts = params.get('num_filts', model.class_emb.in_features)

    # Check if bias should be included in the new class_emb layer
    include_bias = params.get('include_bias', model.class_emb.bias is not None)

    # Create the new class_emb layer
    class_emb = nn.Linear(num_filts, len(taxa_map), bias=include_bias)

    # Initialize the new layer's weights and biases (if applicable)
    class_emb.weight.data.zero_()
    if include_bias:
        class_emb.bias.data.zero_()

    # Replace the old class_emb layer with the new one
    model.class_emb = class_emb

    # Update the relevant parameters
    params['num_classes'] = len(taxa_map)
    params['class_to_taxa'] = list(taxa_map.keys())

    return model, params


def load_fcnet(args):
    # load model
    net_params = torch.load(args['model_path'], map_location='cpu')
    params = net_params['params']
    params['device'] = args['device']
    model_name = models.select_model(params['model'])
    model = model_name(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                       num_filts=params['num_filts'], num_users=params['num_users'],
                       num_context=params['num_context']).to(params['device'])
    model.load_state_dict(net_params['state_dict'])
    model.eval()
    # I am doing this so that when I save the model, the params will reflect the new model and not the old one
    # Additionally having these parameters set correctly makes it much easier to use certain functions
    # that already exist in our code
    params['env_loc'] = args['env_loc']
    params['seed'] = args['seed']
    params['batch_size'] = args['batch_size']
    return model, params


def load_linearnet(args):
    # load model
    net_params = torch.load(args['model_path'], map_location='cpu')
    params = net_params['params']
    params['device'] = args['device']
    params['num_filts'] = params['num_feats']
    model_name = models.select_model(params['model'])
    model = model_name(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                       num_filts=params['num_filts'], num_users=params['num_users'],
                       num_context=params['num_context']).to(params['device'])
    model.load_state_dict(net_params['state_dict'])
    model.eval()
    # I am doing this so that when I save the model, the params will reflect the new model and not the old one
    # Additionally having these parameters set correctly makes it much easier to use certain functions
    # that already exist in our code
    params['env_loc'] = args['env_loc']
    params['seed'] = args['seed']
    params['batch_size'] = args['batch_size']
    return model, params

def load_model(args):
    if args['use_linear_model'] == 0:
        model, params = load_fcnet(args)
    if args['use_linear_model'] == 1:
        model, params = load_linearnet(args)
        params['num_filts'] = params['num_feats']
        # model = models.LinearNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
        #                    num_filts=params['num_filts'], num_users=params['num_users'],
        #                    num_context=params['num_context']).to(params['device'])
    return model, params
import torch
import torch.utils.data
import torch.nn as nn


def select_model(model_name):
    if model_name == 'FCNet':
        return FCNet
    if model_name == 'LinearNet':
        return LinearNet
    else:
        print('Invalid model specified')
        return None


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=0, num_context=0, include_bias=0):
        super(FCNet, self).__init__()
        if include_bias == 0:
            self.inc_bias=False
        else:
            self.inc_bias=True
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)
        self.context_emb = nn.Linear(num_filts, num_context, bias=self.inc_bias)

        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))

    def forward(self, x, class_of_interest=None, return_feats=False, use_feats_as_input=False, class_mask=None,
                return_logits=False):
        if use_feats_as_input:
            loc_emb = x
            if class_of_interest is None:
                if class_mask is not None:
                    class_pred = self.class_masked_eval(loc_emb, class_mask)
                else:
                    class_pred = self.class_emb(loc_emb)
            else:
                class_pred = self.eval_single_class(loc_emb, class_of_interest)

        else:
            loc_emb = self.feats(x)
            if return_feats:
                return loc_emb
            if class_of_interest is None:
                if class_mask is not None:
                    class_pred = self.class_masked_eval(loc_emb, class_mask)
                else:
                    class_pred = self.class_emb(loc_emb)
            else:
                class_pred = self.eval_single_class(loc_emb, class_of_interest)

        if return_logits:
            return class_pred
        else:
            return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])

    def class_masked_eval(self, x, class_mask):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_mask].T) + self.class_emb.bias[class_mask]
        else:
            return torch.matmul(x, self.class_emb.weight[class_mask].T)


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=0, num_context=0, include_bias=0):
        super(LinearNet, self).__init__()
        if include_bias == 0:
            self.inc_bias = False
        else:
            self.inc_bias = True
        self.class_emb = nn.Linear(num_inputs, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_inputs, num_users, bias=self.inc_bias)
        self.context_emb = nn.Linear(num_inputs, num_context, bias=self.inc_bias)

        self.feats = nn.Identity()  # does not do anything

    def forward(self, x, class_of_interest=None, return_feats=False, use_feats_as_input=False, class_mask=None,
                return_logits=False):
        if use_feats_as_input:
            loc_emb = x
            if class_of_interest is None:
                if class_mask is not None:
                    class_pred = self.class_masked_eval(loc_emb, class_mask)
                else:
                    class_pred = self.class_emb(loc_emb)
            else:
                class_pred = self.eval_single_class(loc_emb, class_of_interest)

        else:
            loc_emb = self.feats(x)
            if return_feats:
                return loc_emb
            if class_of_interest is None:
                if class_mask is not None:
                    class_pred = self.class_masked_eval(loc_emb, class_mask)
                else:
                    class_pred = self.class_emb(loc_emb)
            else:
                class_pred = self.eval_single_class(loc_emb, class_of_interest)

        if return_logits:
            return class_pred
        else:
            return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])

    def class_masked_eval(self, x, class_mask):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_mask].T) + self.class_emb.bias[class_mask]
        else:
            return torch.matmul(x, self.class_emb.weight[class_mask].T)

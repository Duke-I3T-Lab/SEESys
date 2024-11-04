import numpy as np
from torch.utils.data import Dataset
import torch

from . import augmentation as aug
import random

import ptwt
import pywt
import torch.nn.functional as F


def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)
    #print(lengths)
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, IDs



def augment_feature_with_prob(X, prob):
    if(random.random()<prob):
        X = X.reshape((1, X.shape[0], X.shape[1]))
        X = aug.window_warp(X, 0.1, 3)
        X = aug.jitter(X, 0.1)
        X = np.squeeze(X)
    return X

def get_step_sliding_windows(indices, step):
    if(step>1):
        latest_index = indices[0]
        step_indices = [latest_index]
        for i, index in enumerate(indices):
            if(index-latest_index >= step):
                step_indices.append(index)
                latest_index = index
        return step_indices
    else:
        return indices


def interpolate(X, length):
    '''
    X is a tensor with shape (channel, length)
    '''
    X = X.unsqueeze(0)
    X = F.interpolate(X, size=length, mode='linear', align_corners=False)
    return X.squeeze(0)


def DWTtransform(X, wavelet, level):
    """
    (seq_length, feat_dim) array in tensor
    """
    # permute the tensor to (feat_dim, seq_length)
    X = X.permute(1,0)
    length = int(X.shape[1]/2)
    X_DWT = ptwt.wavedec(X, wavelet, mode='constant', level=level)
    X_DWT = torch.cat([interpolate(x, length) for x in X_DWT], dim=0)
    return X_DWT.permute(1,0) # permute back to (seq_length, feat_dim)


class ClassiregressionDataset(Dataset):

    def __init__(self, data, indices, aug=False, prob=0.1, step=1, enableDWT=True, wavelet='haar', level=2):
        super(ClassiregressionDataset, self).__init__()

        self.aug = aug
        self.prob = prob
        self.enableDWT = enableDWT
        if(self.enableDWT):
            self.wavelet = pywt.Wavelet(wavelet)
            self.level = level


        self.data = data  # this is a subclass of the BaseData class in data.py

        indices = get_step_sliding_windows(indices, step)

        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]

        self.labels_df = self.data.labels_df.loc[self.IDs]


    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        if(self.aug):
            X = augment_feature_with_prob(X, self.prob)
        X = torch.from_numpy(X)
        if(self.enableDWT):
            X =  DWTtransform(X, self.wavelet, self.level)

        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array
        y = torch.from_numpy(y)
        return X, y, self.IDs[ind]

    def __len__(self):
        return len(self.IDs)



def transduct_mask(X, mask_feats, start_hint=0.0, end_hint=0.0):
    """
    Creates a boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        mask_feats: list/array of indices corresponding to features to be masked
        start_hint:
        end_hint: proportion at the end of time series which will not be masked

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """

    mask = np.ones(X.shape, dtype=bool)
    start_ind = int(start_hint * X.shape[0])
    end_ind = max(start_ind, int((1 - end_hint) * X.shape[0]))
    mask[start_ind:end_ind, mask_feats] = 0

    return mask


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def collate_unsuperv(data, max_len=None, mask_compensation=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X,
                                    dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks, IDs


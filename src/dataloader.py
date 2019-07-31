"""Data loading utilies."""

import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torchvision

import albumentations as A

from constants import *


def aug_img(im, transform):
  """Augment the image stack."""
  im = np.transpose(im, [1, 2, 0])
  im = transform(image=im)['image']
  im = np.transpose(im, [2, 0, 1])
  return im


def normalize(vol, rgb=True, transform=None):
  pad = int((vol.shape[2] - INPUT_DIM)/2)
  if pad != 0:
    vol = vol[:,pad:-pad,pad:-pad]

  if transform:
    vol = aug_img(vol, transform)

  vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
  # normalize
  vol = (vol - MEAN) / STDDEV
  # convert to RGB
  if rgb:
    vol = np.stack((vol,)*3, axis=1)
  else:
    vol = np.expand_dims(vol, 1)
  return vol


class Dataset(data.Dataset):

  def __init__(self, data_dir, meta, rgb=True, transform=None, cat='all'):
    super().__init__()
    self.meta = meta
    self.data_dir = data_dir
    if cat == 'all':
      self.category = ['abnormal', 'acl', 'meniscus']
    else:
      self.category = [cat]
    self.img_type = ['axial', 'coronal', 'sagittal']
    self.rgb = rgb
    self.transform = transform


  def __getitem__(self, index):
    row = self.meta.iloc[index]
    data_item = {}
    for im_type in self.img_type:
      path = os.path.join(self.data_dir, row['sub_dir'],
                          im_type, row['Case'] + '.npy')
      with open(path, 'rb') as f:
        vol = np.load(f).astype(np.float32)
        data_item[im_type] = normalize(vol, self.rgb, self.transform)

    label = row[self.category].values.astype(np.float32)

    return {'data': data_item, 'label': label}

  def __len__(self):
    return self.meta.shape[0]


def get_data_loader(data_dir, meta, shuffle=True, rgb=True, in_memory=False,
                    transform=None, cat='all'):
  dataset = Dataset(data_dir, meta, rgb=rgb, transform=transform, cat=cat)
  loader = data.DataLoader(dataset, batch_size=1, num_workers=4,
      worker_init_fn=lambda x: np.random.seed(
        int(torch.initial_seed() + x) % (2 ** 32 - 1)),
      shuffle=shuffle)
  return loader

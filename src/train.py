"""Function to train the models."""
from datetime import datetime
import importlib
from pathlib import Path
import os
import random
import re
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

import torch

from dataloader import get_data_loader


def read_meta(data_path, meta_type='train'):
  """Function to read the meta data."""

  path = data_path + '%s/' % meta_type
  category = ['abnormal', 'acl', 'meniscus']
  meta_list = []
  for cat in category:
    meta = pd.read_csv(data_path + '/%s-%s.csv' % (meta_type, cat),
        header=None, names=['Case', cat],
        dtype={'Case': str, cat: np.int64}).set_index('Case')
    meta_list.append(meta)
  meta_df = pd.concat(meta_list, axis=1)
  meta_df['sub_dir'] = meta_type
  return meta_df.reset_index()


def to_device(data, device):
  if isinstance(data, dict):
    return {k: v.to(device) for k, v in data.items()}
  else:
    return data.to(device)


def run_model(model, criterion, loader, epoch, train=False,
              optimizer=None, use_gpu=True, im_type='axial',
              label_smoothing=0):
  preds = []
  labels = []

  if train:
    model.train()
  else:
    model.eval()

  total_loss = 0.
  num_batches = 0

  for batch in loader:
    if train:
      optimizer.zero_grad()

    vol, label = batch['data'], batch['label']
    if use_gpu:
      vol = to_device(vol, 'cuda:0')
      label = to_device(label, 'cuda:0')

    pred = model(vol, im_type)
    if label_smoothing > 0:
      loss = criterion(pred, label * (1 - 2 * label_smoothing) +
          label_smoothing)
    else:
      loss = criterion(pred, label)
    total_loss += loss.item()

    pred_npy = pred.data.cpu().numpy()
    label_npy = label.data.cpu().numpy()

    preds.append(pred_npy)
    labels.append(label_npy)

    if train:
      loss.backward()
      optimizer.step()
    num_batches += 1

  avg_loss = total_loss / num_batches

  preds_np = np.concatenate(preds, axis=0)
  labels_np = np.concatenate(labels, axis=0)

  auc_list = []
  for i in range(preds_np.shape[1]):
    fpr, tpr, threshold = metrics.roc_curve(labels_np[:, i], preds_np[:, i])
    auc = metrics.auc(fpr, tpr)
    auc_list.append(auc)

  return avg_loss, np.mean(auc_list), auc_list, preds_np, labels_np


def evaluate(model, loader, n_round=3, use_gpu=True, im_type='axial'):

  model.eval()

  prediction_list = []
  for i in range(n_round):
    preds = []
    labels = []
    for batch in loader:
      vol, label = batch['data'], batch['label']
      if use_gpu:
        vol = to_device(vol, 'cuda:0')
        label = to_device(label, 'cuda:0')

      with torch.no_grad():
        pred = model(vol, im_type)
      pred_npy = pred.data.cpu().numpy()
      label_npy = label.data.cpu().numpy()

      preds.append(pred_npy)
      labels.append(label_npy)

    preds_np = np.concatenate(preds, axis=0)
    prediction_list.append(preds_np)
    labels_np = np.concatenate(labels, axis=0)
  avg_pred = np.mean(np.stack(prediction_list, axis=0), axis=0)
  auc_list = []

  for i in range(avg_pred.shape[1]):
    fpr, tpr, threshold = metrics.roc_curve(labels_np[:, i], avg_pred[:, i])
    auc = metrics.auc(fpr, tpr)
    auc_list.append(auc)
  print('mean auc ', np.mean(auc_list))
  print('individual auc ', auc_list)

  return avg_pred, labels_np


def log(log_file, mesgs):
  if not isinstance(mesgs, list):
    mesgs = [mesgs]
  with open(log_file, 'a') as f:
    for mesg in mesgs:
      print(mesg)
      f.write(mesg + '\n')


def train(config):

  train_meta = read_meta(data_path=config['data_dir'], meta_type='train')
  val_meta = read_meta(data_path=config['data_dir'], meta_type='valid')

  cat = 'all'
  if 'cat' in config:
    cat = config['cat']
  train_loader = get_data_loader(config['data_dir'], train_meta,
                                 shuffle=True, rgb=config['rgb'],
                                 transform=config['transform'], cat=cat)
  val_transform = None
  if 'val_transform' in config:
    val_transform = config['val_transform']
  elif 'transform' in config:
    val_transform = config['transform']

  val_loader = get_data_loader(config['data_dir'], val_meta,
                               shuffle=False, rgb=config['rgb'],
                               transform=val_transform, cat=cat)

  model = config['model']

  if config['use_gpu']:
    model = model.cuda()

  criterion = torch.nn.BCELoss()

  best_val_loss = float('inf')
  best_val_auc = 0
  best_file_name = None
  log_file = Path(config['run_dir']) / 'train_log.txt'
  log(log_file, 'Training model with run_dir %s' % config['run_dir'])

  start_time = datetime.now()

  for epoch in range(config['epochs']):
    optimizer = config['optimizer']
    scheduler = config['scheduler']

    change = datetime.now() - start_time
    log(log_file,
        'starting epoch {}. time passed: {}'.format(epoch+1, str(change)))

    label_smoothing = 0
    if 'label_smoothing' in config:
      label_smoothing = config['label_smoothing']

    train_loss, train_auc, train_auc_list, _, _ = run_model(
        model, criterion, train_loader,
        epoch, train=True, optimizer=optimizer,
        im_type=config['im_type'], label_smoothing=label_smoothing)

    log(log_file, f'  train loss: {train_loss:0.4f}')
    log(log_file, f'  train AUC: {train_auc:0.4f}')
    train_auc_output = ': '.join(['%0.4f' % x for x in train_auc_list])
    log(log_file, f'  train AUC individual: {train_auc_output}')

    # Running the validation set
    val_loss, val_auc, val_auc_list, _, _,  = run_model(
        model, criterion, val_loader,
        epoch, im_type=config['im_type'],
        label_smoothing=label_smoothing)

    log(log_file, f'  valid loss: {val_loss:0.4f}')
    log(log_file, f'  valid AUC: {val_auc:0.4f}')
    val_auc_output = ': '.join(['%0.4f' % x for x in val_auc_list])

    log(log_file, f'  val AUC individual: {val_auc_output}')

    scheduler.step(val_loss)

    if val_loss < best_val_loss or val_auc > best_val_auc:
      if val_loss < best_val_loss:
        best_val_loss = val_loss
      if val_auc > best_val_auc:
        best_val_auc = val_auc

      if config['save_model']:
        file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}'
        '_auc{val_auc:0.4f}_epoch{epoch+1}'
        save_path = Path(config['run_dir']) / file_name
        torch.save(model.state_dict(), save_path)
        best_file_name = file_name
    if 'call_back' in config:
      config['call_back'](epoch, config, train_loss,
          train_auc, val_loss, val_auc)


def run_eval(config):

  val_meta = read_meta(data_path=config['data_dir'], meta_type='valid')

  cat = 'all'
  if 'cat' in config:
    cat = config['cat']
  val_loader = get_data_loader(config['data_dir'], val_meta,
                               shuffle=False, rgb=config['rgb'],
                               transform=config['transform'], cat=cat)

  model = config['model']
  state_dict = torch.load(config['model_path'],
      map_location=(None if config['use_gpu'] else 'cpu'))
  model.load_state_dict(state_dict)

  if config['use_gpu']:
    model = model.cuda()
  evaluate(model, val_loader, config['n_round'],
      config['use_gpu'], config['im_type'])


def set_seed(seed=1029, use_gpu=False):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if use_gpu:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def run_exp(config):
  print("########################################")
  print("running experiment with config")
  print(config)
  print("########################################")
  set_seed(config['seed'], config['use_gpu'])

  os.makedirs(config['run_dir'], exist_ok=True)

  train(config)


def main():
  exp = sys.argv[1]
  exp_file = exp.replace('/', '.')
  exp_file = re.sub('\.py$', '', exp_file)

  exp_module = importlib.import_module(exp_file)
  config = exp_module.config
  run_exp(config)

if __name__ == '__main__':
  main()

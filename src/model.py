"""Construct the model from layers."""
import torch.nn as nn
import torchvision
from layers import *


def cnn_backbone():
  """CNN feature extraction model."""
  backbone = torchvision.models.vgg11(pretrained=True).features
  backbone_module = torch.nn.Sequential(
      FeatureWrapper(backbone),
      Sgate(),
      nn.Linear(512, 256)
      )
  return backbone_module


def create_model():

  feature_module1 = FeatureNet(cnn_backbone())
  feature_module2 = FeatureNet(cnn_backbone())
  feature_module3 = FeatureNet(cnn_backbone())

  model = TriNet(feature_module1, feature_module2,
      feature_module3, indim=512)
  return model

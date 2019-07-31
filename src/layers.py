import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv1d(f1, f2, kw, dw, pw, dilation=1):
  """1D convolution wrapper."""
  layer  = nn.Conv1d(f1, f2, kw, dw, pw, dilation=dilation)
  nl = kw * f1
  # He initialization strategy used in ResNet
  layer.weight.data.normal_(0, math.sqrt(2. / nl))
  layer.bias.data.zero_()
  return layer


class DilateConv(nn.Module):
  """Dilated convolution used in the WaveNet module."""
  def __init__(self, f1, f2, kw, dw, pw, diw, dropout=0):
    super(DilateConv, self).__init__()
    if type(pw) == tuple:
      self.conv1 = conv1d(f1, 2 * f2, kw, dw, 0, dilation=diw)
      self.pad = True
      self.left_pad, self.right_pad = pw
    else:
      self.pad = False
      self.conv1 = conv1d(f1, 2 * f2, kw, dw, pw, dilation=diw)

    self.f1 = f1
    self.f2 = f2
    self.dropout = nn.Dropout(dropout)
    if f1 != f2:
      self.conv2 = conv1d(f1, f2, 1, 1, 0)

  def forward(self, x):
    if self.pad:
      out = F.pad(x, (self.left_pad, self.right_pad), 'constant', 0)
      conv1_out = self.conv1.forward(out)
    else:
      conv1_out = self.conv1.forward(x)
    out1 = torch.sigmoid(conv1_out.narrow(1, 0, self.f2))
    out2 = torch.tanh(conv1_out.narrow(1, self.f2, self.f2))
    out = out1 * out2
    if self.f1 == self.f2:
      out = self.dropout(x) + out
    else:
      out1 = self.conv2.forward(x)
      out1 = F.leaky_relu(out1, 0.1)
      out = self.dropout(out1) + out
    return out


def WaveNet(n, inDim, nextDim, causal=False):
  assert n > 0, "n must be greater than 0"
  module_list = []
  for i in range(n-1):
    if causal:
      module_list.append(DilateConv(inDim, nextDim, 3, 1,
        (2 * 2 ** i, 0), 2 ** i))
    else:
      module_list.append(DilateConv(inDim, nextDim, 3, 1, 2 ** i, 2 ** i))
  return nn.Sequential(*module_list)


class Sgate(nn.Module):
  """swish activation."""
  def __init__(self):
    super(Sgate, self).__init__()

  def forward(self, x):
    return x * torch.sigmoid(x)


class FeatureWrapper(nn.Module):
  """For alex and vgg like network that has features param."""
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.pool = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    x = self.model(x)
    x = self.pool(x).view(x.shape[0], -1)
    return x


class FeatureNet(nn.Module):
  def __init__(self, backbone):
    super().__init__()
    self.model = backbone
    self.wavenet_module = nn.Sequential(
      WaveNet(3, 256, 256),
      conv1d(256, 256, 1, 1, 0),
      )
    self.w_layer = conv1d(256, 1, 1, 1, 0)

  def forward(self, x):
    n, s, c, h, w = x.shape
    x = x.reshape(-1, c, h, w)
    x = self.model(x)
    x = x * torch.sigmoid(x)
    x = x.reshape(n, s, -1)
    x = x.permute(0, 2, 1)

    x = self.wavenet_module(x)
    w = F.softmax(self.w_layer(x), dim=2)
    x1 = torch.max(x, dim=2)[0]
    x = x * w
    x = torch.sum(x, 2, keepdim=False)
    return torch.cat([x, x1], dim=1)


class TriNet(nn.Module):
  """Use all the three image modes."""
  def __init__(self, net1, net2, net3, indim=256, nclass=3):
    super().__init__()
    self.net1 = net1
    self.net2 = net2
    self.net3 = net3
    self.classifier = nn.Linear(indim * 3, nclass)

  def forward(self, x, *arg):
    axial_out = self.net1(x['axial'])
    coronal_out = self.net2(x['coronal'])
    sag_out = self.net3(x['sagittal'])
    out = torch.cat([axial_out, coronal_out, sag_out], dim=1)
    out = self.classifier(out)
    out = torch.sigmoid(out)
    return out

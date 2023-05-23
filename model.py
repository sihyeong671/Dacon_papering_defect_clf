from transformers import ViTForImageClassification, ConvNextForImageClassification, SwinForImageClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class BaseModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = timm.create_model("efficientnet_b4", pretrained=True)
    self.model.classifier = nn.Linear(in_features=1792, out_features=19, bias=True)
  
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x), None
    

class ViT_base_384(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-384")
    self.model.classifier = nn.Linear(768, 19, bias=True)
  
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x.logits), None
  

class EfficientNetV2_m(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature = timm.create_model("efficientnetv2_m", pretrained=True)
    
    self.conv_head = self.feature.conv_head
    self.bn2 = self.feature.bn2
    self.global_pool = self.feature.global_pool
    self.classifier = nn.Linear(in_features=1280, out_features=19, bias=True)
    
    self.feature.conv_head = nn.Identity()
    self.feature.bn2 = nn.Identity()
    self.feature.global_pool = nn.Identity()
    self.feature.classifier = nn.Identity()
  
  def forward(self, x):
    x = self.feature(x)
    x = self.conv_head(x)
    _map = self.bn2(x)
    x = self.global_pool(_map)
    x = self.classifier(x)
    return F.sigmoid(x), _map


class ResNext_101(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature = timm.create_model("resnext101_64x4d", pretrained=True)
    self.gap = self.feature.global_pool
    self.fc = nn.Linear(in_features=2048, out_features=19, bias=True)
    
    self.feature.global_pool = nn.Identity()
    self.feature.fc = nn.Identity()
  
  def forward(self, x):
    _map = self.feature(x)
    x = self.gap(_map)
    x = self.fc(x)
    return x, _map


class DenseNet_201(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = timm.create_model("densenet201", pretrained=True)
    self.model.classifier = nn.Linear(in_features=1920, out_features=19, bias=True)
    
  def forward(self, x):
    x = self.model(x)
    return x, None
  

class ConvNext_tiny(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
    self.model.classifier = nn.Linear(in_features=768, out_features=19, bias=True)
    
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x.logits), None
  
  
class SwinBase(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    self.model.classifier = nn.Linear(in_features=1024, out_features=19, bias=True)

  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x.logits), None
  
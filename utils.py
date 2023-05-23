import os
import random
from glob import glob

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import timm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import albumentations as A
from albumentations.pytorch import ToTensorV2 

from model import *

class Config:
  def __init__(self, args):
    self.SEED = args.seed
    self.EPOCHS = args.epochs
    self.LR = args.lr
    self.IMG_RESIZE = args.img_resize
    self.NUM_WORKERS = args.num_workers
    self.BATCH_SIZE = args.batch_size
    self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.LABELCODER = LabelCoder()
    self.MODE = args.mode
    self.MODEL_NAME = args.model_name
    # self.detail = args.detail


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CDBLoss(nn.Module):
  def __init__(self, class_difficulty, device="cpu", tau="dynamic", reduction="mean"):
    super().__init__()
    
    if tau == "dynamic":
      bias = (1 - np.min(class_difficulty)) / (1 - np.max(class_difficulty) + 0.01)
      tau = self._sigmoid(bias)
    else:
      tau = float(tau)
    
    weights = class_difficulty ** tau
    # normalize?
    weights = weights / weights.sum() * len(weights)
    self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights), reduction=reduction).to(device)
    
  def forward(self, input, target):
    return self.loss_fn(input, target)
  
  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

class CustomDataset(Dataset):
  def __init__(self, img_paths, label_lst, transforms=None):
    self.img_paths = img_paths
    self.label_lst = label_lst
    self.transforms = transforms
    
  def __getitem__(self, index):
    path = self.img_paths[index]
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if self.transforms is not None:
      img = self.transforms(image=img)["image"]
    
    if self.label_lst is not None:
      label = self.label_lst[index]
      return img, label
    else:
      return img
    
  def __len__(self):
    return len(self.img_paths)
  

class LabelCoder():
  def __init__(self):
    all_img_list = glob("./dataset\\train\\*\\*")
    df = pd.DataFrame(columns=["img_path", "label"])
    df["img_path"] = all_img_list 
    df["label"] = df["img_path"].apply(lambda x: str(x).split("\\")[-2])
    self.le = preprocessing.LabelEncoder()
    self.le.fit(df["label"])
    del df
    
  def encoding(self, label):
    return self.le.transform(label)
  
  def decoding(self, label):
    return self.le.inverse_transform(label)
  
  def get_label(self):
    return self.le.classes_
  
  
class DataModule(pl.LightningDataModule):
  def __init__(self, CONFIG: Config):
    super().__init__()
    self.CONFIG = CONFIG
    
  
  def setup(self, stage=None):
    train_transform = A.Compose([
      A.Resize(self.CONFIG.IMG_RESIZE, self.CONFIG.IMG_RESIZE),
      A.HorizontalFlip(),
      A.OneOf([
        A.MotionBlur(p=1),
        A.Blur(p=1),
        A.GaussianBlur(p=1),
        A.Defocus(p=1),
        A.RandomBrightnessContrast(p=1),
        A.CoarseDropout(p=1)
      ]),
      A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
      ToTensorV2()
    ])
    
    test_transform = A.Compose([
      A.Resize(self.CONFIG.IMG_RESIZE, self.CONFIG.IMG_RESIZE),
      A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
      ToTensorV2()
    ])
    
    all_img_list = glob(".\\dataset\\train\\*\\*") # os: window
    df = pd.DataFrame(columns=["img_path", "label"])
    df["img_path"] = all_img_list
    df["label"] = df["img_path"].apply(lambda x: str(x).split("\\")[-2])
    train, val, _, _ = train_test_split(df, df["label"], stratify=df["label"], test_size=0.1, random_state=self.CONFIG.SEED)
    # label encoding
    # df["label"] = self.CONFIG.LABELCODER.encoding(df["label"])
    train["label"] = self.CONFIG.LABELCODER.encoding(train["label"])
    val["label"] = self.CONFIG.LABELCODER.encoding(val["label"])
    
    # made sampler
    # class_count = train["label"].value_counts().tolist()
    # num_samples = sum(class_count)
    # labels = train["label"].values.tolist()
    # class_weights = [num_samples / class_count[i] for i in range(len(class_count))]
    # weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    # self.sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    
    
    self.train_dataset = CustomDataset(
      img_paths=train["img_path"].values,
      label_lst=train["label"].values,
      transforms=train_transform
    )
    
    self.val_dataset = CustomDataset(
      img_paths=val["img_path"].values,
      label_lst=val["label"].values,
      transforms=test_transform
    )
    
    test_df = pd.read_csv(".\\dataset\\test.csv")
    test_df["img_path"] = test_df["img_path"].apply(lambda x: os.path.join(".\\dataset", x))
    self.test_dataset = CustomDataset(
      img_paths=test_df["img_path"].values,
      label_lst=None,
      transforms=test_transform
    )
  
  def train_dataloader(self):
    
    return DataLoader(
      dataset=self.train_dataset,
      shuffle=True, # sampler 와 동시에 사용 불가
      num_workers=self.CONFIG.NUM_WORKERS,
      batch_size=self.CONFIG.BATCH_SIZE,
      # sampler=self.sampler
    )
  
  def val_dataloader(self):
    return DataLoader(
      dataset=self.val_dataset,
      shuffle=False,
      num_workers=self.CONFIG.NUM_WORKERS,
      batch_size=self.CONFIG.BATCH_SIZE
    )
  
  def test_dataloader(self):
    return DataLoader(
      dataset=self.test_dataset,
      shuffle=False,
      num_workers=self.CONFIG.NUM_WORKERS,
      batch_size=self.CONFIG.BATCH_SIZE
    )
  
  def predict_dataloader(self):
    return DataLoader(
      dataset=self.train_dataset,
      shuffle=False,
      num_workers=self.CONFIG.NUM_WORKERS,
      batch_size=self.CONFIG.BATCH_SIZE
    )

    
class LightningModel(pl.LightningModule):
  def __init__(self, CONFIG: Config):
    super().__init__()

    self.CONFIG = CONFIG
    # CONFG에 따라서 모델 가져오는 코드 작성
    # assert self.CONFIG.MODEL_NAME is not None
    # try:
    #   module = importlib.import_module("model")
    #   self.model = getattr(module, self.CONFIG.MODEL_NAME)
    # except AttributeError:
    #   print(f"{self.CONFIG.MODEL_NAME}과 일치하는 모델이 없습니다")
    self.model = eval(f"{self.CONFIG.MODEL_NAME}()")
    # if 문 걸어주기
    # self.loss_fn = CDBLoss(class_difficulty = np.ones(19), device=CONFIG.DEVICE).to(CONFIG.DEVICE)
    self.loss_fn = nn.CrossEntropyLoss()
    self.acc_metric = MulticlassAccuracy(num_classes=19, average='none')
    self.f1_metric = MulticlassF1Score(num_classes=19) # average = None
    # self.train_step_outputs = []
    self.val_loss_list = []
    self.val_acc_list = []
    self.val_f1_list = []
    self.test_step_outputs = []
    
  def forward(self, x):
    x, _map = self.model(x)
    return x, _map
  
  def training_step(self, batch, batch_idx):
    loss, acc, f1_score = self._common_step(batch, batch_idx,)
    # self.train_step_outputs.append(loss)
    metrics = {"train_loss": loss, "train_acc": acc, "train_f1_score": f1_score}
    self.log_dict(metrics)
    return loss
  
  # def on_train_epoch_end(self):
  #   self.train_step_outputs.clear()
  
  def validation_step(self, batch, batch_idx):
    loss, acc, f1_score = self._common_step(batch, batch_idx, mode='val')
    # save each metric
    # 이 코드를 더 간단하게 할 수는 없나?
    self.val_loss_list.append(loss)
    self.val_acc_list.append(acc)
    self.val_f1_list.append(f1_score)
  
  def on_validation_epoch_start(self):
    self.confusion_metric = MulticlassConfusionMatrix(num_classes=19).to(self.device)
    
  def on_validation_epoch_end(self):
    
    # confusion matrix 저장
    cmf = self.confusion_metric.compute().cpu().numpy()
    df_cmf = pd.DataFrame(
      cmf,
      index=self.CONFIG.LABELCODER.get_label(),
      columns=self.CONFIG.LABELCODER.get_label()
    )
    f, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df_cmf, annot=True, ax=ax)
    
    metrics = {
      "val_loss": self._get_avg(self.val_loss_list),
      "val_acc": self._get_avg(self.val_acc_list),
      "val_f1_score": self._get_avg(self.val_f1_list),
    }
    self.logger.experiment.log({
      "confusion_matrix": wandb.Image(f)
    })
    self.log_dict(metrics)
    
    # memory clear
    self.val_loss_list.clear()
    self.val_acc_list.clear()
    self.val_f1_list.clear()
    plt.close()
    
  def _common_step(self, batch, batch_idx, mode='train'):
    img, label = batch
    label = label.type(torch.LongTensor).to(self.CONFIG.DEVICE)
    output, _ = self(img)
    acc = self.acc_metric(output, label) # class wise acc
    loss = self.loss_fn(output, label)
    f1_score = self.f1_metric(output, label)
    
    if mode == 'val':
      self.confusion_metric.update(output, label) # 값 누적
      # if 문 걸어주기
      # difficulty = 1 - acc
      # self.loss_fn = CDBLoss(class_difficulty=difficulty.detach().cpu().numpy(), device=self.CONFIG.DEVICE)
    
    acc = acc.mean()
    return loss, acc, f1_score
    
  def test_step(self, batch, batch_idx):
    img = batch
    output, _ = self(img)
    self.test_step_outputs += (output.argmax(1).detach().cpu().numpy().tolist())
  
  def on_test_epoch_end(self):
    preds = self.CONFIG.LABELCODER.decoding(self.test_step_outputs)
    submit = pd.read_csv(".\\dataset\\sample_submission.csv")
    submit["label"] = preds
    submit.to_csv(f".\\{self.CONFIG.MODEL_NAME}.csv", index=False)
  
  def predict_step(self, batch, batch_idx: int):
    # CAM 이미지 저장
    imgs, labels = batch
    labels = labels.view(labels.size(0), -1) # B -> B 1
    _, _map = self(imgs)
    weight_fc = self.model.fc.weight.data.T
    
    W = torch.stack([weight_fc[:, labels[i]] for i in range(len(labels))])
    W = W.unsqueeze(dim=-1)
    cam = torch.mul(_map, W)
    cam = torch.sum(cam, dim=1) # B h' w'
    cam = cam.detach().cpu().numpy()
    
    for i in range(len(batch)):
      fig, ax = plt.subplots(figsize=(15, 15))
      origin = imgs[i].detach().cpu().numpy()
      origin = origin * 0.5 + 0.5 # denorm
      origin = np.transpose(origin, (1, 2, 0))
      label = labels[i].detach().cpu().numpy()
      label = self.CONFIG.LABELCODER.decoding(label)[0]
      final_cam = cv2.resize(cam[i], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
      ax.imshow(origin)
      ax.imshow(final_cam, alpha=0.4, cmap="jet")
      fig.savefig(f"./camp/{batch_idx}_{i}_{label}.png")
      plt.close() # 생성된 plot figure가 메모리를 점유하고 있기 때문에 저장후 삭제 필요
    
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.CONFIG.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
    )
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": scheduler,
        "monitor" : "val_loss"
      }
    }

  
  def _get_avg(self, metric_list):
    return sum(metric_list) / len(metric_list)
import argparse

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import (
  DataModule,
  LightningModel,
  Config,
  seed_everything,
)


def train(CONFIG: Config):
  data_module = DataModule(CONFIG)
  data_module.setup()
  
  model = LightningModel(CONFIG=CONFIG)
  
  ckpt = ModelCheckpoint(
    save_top_k=1,
    monitor="val_f1_score", # val f1
    mode="max",
    dirpath=".\\ckpt",
    filename="ViT_base_384_v2_{epoch}"
  )
  
  wandb_logger = WandbLogger(
    entity="bsh",
    name=f"{CONFIG.MODEL_NAME}_v2",
    project="dacon_clf", 
  )
  
  trainer = pl.Trainer(
    max_epochs=CONFIG.EPOCHS,
    accelerator="gpu",
    callbacks=ckpt,
    logger=wandb_logger
  )
  
  trainer.fit(
    model,
    data_module
  )
      
def test(CONFIG: Config):
  data_module = DataModule(CONFIG)
  data_module.setup()
  
  model = LightningModel.load_from_checkpoint(".\\ckpt\\ViT_base_384_v2_epoch=6.ckpt", CONFIG=CONFIG)
  
  trainer = pl.Trainer(
    accelerator="gpu",
  )

  trainer.test(
    model,
    data_module
  )
  
  
def plot_cam(CONFIG: Config):
  data_module = DataModule(CONFIG)
  data_module.setup()
  
  #
  model = LightningModel.load_from_checkpoint(".\\ckpt\\ViT_base_384_v2_epoch=6.ckpt", CONFIG=CONFIG)
  
  trainer = pl.Trainer(
    precision=16,
    accelerator="gpu",
  )
  
  trainer.predict(
    model,
    data_module
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=777)
  parser.add_argument('--epochs', type=int, default=30)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--img_resize', type=int, default=384)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--mode', default='train')
  parser.add_argument('--model_name', default="ViT_base_384")
  # parser.add_argument('--detail') 
  args = parser.parse_args()
  
  CONFIG = Config(args=args)
  plt.rcParams["font.family"] = "MalGun Gothic"
  
  # seed everything
  seed_everything(CONFIG.SEED)
  
  
  if CONFIG.MODE == 'train':
    train(CONFIG)
  elif CONFIG.MODE == 'test':
    test(CONFIG)
  elif CONFIG.MODE == 'predict':
    plot_cam(CONFIG)
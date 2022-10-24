import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from model.linear import LinearClassifer
from dataset.animal import AnimalDataset


def main():
    # model
    BATCH_SIZE = 1024
    lr_monitor = LearningRateMonitor(logging_interval='step')
    train_dataset = AnimalDataset("train")
    val_dataset = AnimalDataset("test")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=16)
    
    model = LinearClassifer(in_features=train_dataset.NUM_LATENT, out_features=train_dataset.NUM_CLASS)
    # training
    trainer = pl.Trainer(
        gpus=1, 
        precision=32,
        check_val_every_n_epoch=1,
        max_epochs=30,
        callbacks=[lr_monitor]
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
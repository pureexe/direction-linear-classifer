import torch 
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LinearClassifer(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.build_network()
        #self.loss_fn = F.mse_loss
        #taken from https://towardsdatascience.com/linear-classification-in-pytorch-9d8a8f8ff264
        self.loss_fn = F.binary_cross_entropy_with_logits 

    def build_network(self):
        self.net = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=False),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5)
        #          
        #lambda_fn = lambda epoch: 0.1 ** (epoch // 3)
        lambda_fn = lambda epoch: 0.1 ** (epoch // 5)  if epoch < 15 else 0.1 ** 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn, verbose=False)
        return [optimizer], [scheduler]

    def forward(self, x):
        y = self.net(x)
        return y
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch['latent']
        y = train_batch['label']
        x_hat = self(x)
        loss = self.loss_fn(x_hat, y)
        self.log('loss/train', loss)
        self.calcurate_accuracy(x_hat, y, label="_train")
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['latent']
        y = val_batch['label']
        x_hat = self(x)
        val_loss = self.loss_fn(x_hat, y)
        self.log('loss/validation', val_loss)
        self.calcurate_accuracy(x_hat, y)
       

    def calcurate_accuracy(self, x_hat, y, label=""):
        x_hathot = torch.zeros_like(x_hat)
        x_hat_softmax = torch.nn.functional.softmax(x_hat, dim=1)
        x_hathot[:, torch.argmax(x_hat_softmax,dim=1)] = 1
        avg_acc = []
        for i in range(self.out_features):
            inds = y[:,i] == 1
            total_inds = y[inds].shape[0]
            # get distance vector, need to divide by 2 
            distance = torch.sum((x_hathot[inds] - y[inds]) ** 2) / 2.0 
            acc = 1.0 - (distance / total_inds)
            self.log(f'accuracy{label}/{i:02d}', acc)
            avg_acc.append(acc)
        avg_acc = torch.mean(torch.tensor(avg_acc))
        self.log(f'accuracy{label}/total', avg_acc)
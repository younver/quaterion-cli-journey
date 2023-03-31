import os
import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

CHECKPOINTS_DIR = "app/volume/lightning_basics/"
CHECKPOINT_PATH = CHECKPOINTS_DIR + "checkpoint.ckpt"


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


"""
The LightningModule is the full recipe that defines how your nn.Modules interact.

The training_step defines how the nn.Modules interact together.
In the configure_optimizers define the optimizer(s) for your models.
"""
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        # The LightningModule allows you to automatically save all the hyperparameters passed to init simply by calling self.save_hyperparameters().
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

"""
Under the hood, the Lightning Trainer runs the following training loop on your behalf

autoencoder = LitAutoEncoder(Encoder(), Decoder())
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""


""" Define a PyTorch DataLoader which contains your training dataset. """
# Load data sets
transform = transforms.ToTensor()
train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

# initialize data loaders
train_loader = DataLoader(train_set)
test_loader = DataLoader(test_set)
valid_loader = DataLoader(valid_set)

"""
To train the model use the Lightning Trainer which handles
all the engineering and abstracts away all the complexity needed for scale. 
"""
# initialize the model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# EarlyStopping Callback
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")

# initialize the Trainer | saves checkpoints to default_root_dir at every epoch
trainer = pl.Trainer(callbacks=[early_stop_callback], enable_checkpointing=True, default_root_dir=CHECKPOINTS_DIR)

# test the model
trainer.test(model=autoencoder, dataloaders=test_loader)

# train the model | resume training state from ckpt_path
try:
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=CHECKPOINT_PATH)
except FileNotFoundError:
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)

"""
Inside a Lightning checkpoint you’ll find:

- 16-bit scaling factor (if using 16-bit precision training)
- Current epoch
- Global step
- LightningModule’s state_dict
- State of all optimizers
- State of all learning rate schedulers
- State of all callbacks (for stateful callbacks)
- State of datamodule (for stateful datamodules)
- The hyperparameters (init arguments) with which the model was created
- The hyperparameters (init arguments) with which the datamodule was created
- State of Loops

"""
# Once the autoencoder has trained, pull out the relevant weights for your torch nn.Module
checkpoint = torch.load(CHECKPOINT_PATH)
encoder_weights = checkpoint["encoder"]
decoder_weights = checkpoint["decoder"]
print(checkpoint.keys())

# load model from checkpoint along with its weights and hyperparameters
loaded_model = LitAutoEncoder.load_from_checkpoint(CHECKPOINT_PATH)

# The LightningModule also has access to the Hyperparameters
print(loaded_model.learning_rate)

# disable randomness, dropout, etc...
loaded_model.eval()

# predict with the model
y_hat = loaded_model(x)

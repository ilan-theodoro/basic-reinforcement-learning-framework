import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=4e-3)

if __name__ == "__main__":
    tr_dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    val_dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor(), train=False)
    trainer = L.Trainer(max_epochs=100, check_val_every_n_epoch=10)
    model = LitAutoEncoder()
    trainer.fit(model,
                data.DataLoader(tr_dataset, batch_size=128, num_workers=20),
                data.DataLoader(val_dataset, batch_size=128, num_workers=20))

    # show an image and its reconstruction
    import matplotlib.pyplot as plt

    x, y = val_dataset[0]
    x = x.view(1, -1)
    z = model.encoder(x)
    x_hat = model.decoder(z)
    plt.subplot(1, 2, 1)
    plt.imshow(x.view(28, 28).numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(x_hat.view(28, 28).detach().numpy())
    plt.show()
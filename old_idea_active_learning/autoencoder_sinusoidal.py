import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L


class SinusoidalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     #nn.Dropout1d(0.2),
                                     nn.Linear(128, 24))
        self.decoder = nn.Sequential(nn.Conv1d(1, 8, 9, padding=0),
                                     nn.ReLU(),
                                     nn.AvgPool1d(4, stride=4),
                                     nn.Conv1d(8, 8, 9, padding=0),
                                     nn.ReLU(),
                                     nn.AvgPool1d(4, stride=4),
                                     nn.AdaptiveAvgPool1d(29),
                                     nn.Flatten(1),
                                     #nn.Dropout1d(0.2),
                                     nn.Linear(29*8, 64),
                                     nn.ReLU(),
                                     #nn.Linear(64, 28 * 28))
                                     nn.Linear(64, 10))
        self.register_buffer("frequencies", torch.arange(1, 505, dtype=torch.float32) / 100)


    def forward(self, x):
        x = self.get_signal(x)
        x = self.decoder(x)
        return x

    def get_signal(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), 3, -1).unsqueeze(-1)
        sinusoidal_enc = torch.sum(z[:, 0, :] * torch.sin(z[:, 1, :] * self.frequencies) + z[:, 2, :], dim=1, keepdim=True)
        return sinusoidal_enc

    def get_sinusoid(self, x, idx):
        z = self.encoder(x)
        z = z.view(z.size(0), 3, -1).unsqueeze(-1)
        sinusoidal_enc = z[:, 0, :] * torch.sin(z[:, 1, :] * self.frequencies) + z[:, 2, :]
        return sinusoidal_enc[:, idx]


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.ae = SinusoidalAutoEncoder()

    def forward(self, x):
        z = self.ae.get_signal(x)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.ae(x)
        loss = nn.functional.nll_loss(torch.softmax(x_hat, 1), y)#nn.functional.mse_loss(x_hat, x)
        accuracy = torch.sum(torch.argmax(torch.softmax(x_hat, 1), 1) == y) / len(y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.ae(x)
        loss = nn.functional.nll_loss(torch.softmax(x_hat, 1), y)#nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        if batch_idx == 0:
            # get a line by line figure with subplots of 9 images, each in a row
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(9, 1)
            x = x[0:1]
            for k in range(8):
                axs[k].plot(self.ae.frequencies.cpu().numpy(), self.ae.get_sinusoid(x, k).detach().cpu().numpy()[0])
            axs[8].plot(self.ae.frequencies.cpu().numpy(), self.ae.get_signal(x).detach().cpu().numpy()[0,0])
            plt.show()
            #plt.imshow(x[0].view(28, 28).cpu().numpy())
            #plt.show()
            #plt.imshow(x_hat[0].view(28, 28).detach().cpu().numpy())
            #plt.show()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=4e-4)

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
    x_hat = model(x)
    plt.subplot(1, 2, 1)
    plt.imshow(x.view(28, 28).numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(x_hat.view(28, 28).detach().numpy())
    plt.show()
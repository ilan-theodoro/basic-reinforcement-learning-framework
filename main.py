from timeit import timeit

import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L

from autoencoder import LitAutoEncoder

autoencoder = LitAutoEncoder.load_from_checkpoint("lightning_logs/version_19/checkpoints/epoch=99-step=46900.ckpt")

dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor(), train=True)
dataloader = data.DataLoader(dataset, batch_size=len(dataset))

x, y = next(iter(dataloader))

x = x.view(x.size(0), -1).to(autoencoder.device)[:100]
y = y.to(autoencoder.device)[:100]
z = autoencoder(x)

# # show pca plot of z
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# pca = PCA(n_components=2)
# z_pca = pca.fit_transform(z.detach().cpu().numpy())
# plt.scatter(z_pca[:, 0], z_pca[:, 1], c=y, cmap="tab10", alpha=0.5, s=4)
# plt.show()
#
#
# # show umap plot of z
# import umap
#
# z_umap = umap.UMAP().fit_transform(z.detach().cpu().numpy())
# plt.scatter(z_umap[:, 0], z_umap[:, 1], c=y, cmap="tab10", alpha=0.5, s=4)
# plt.show()

# compute linear classification with normal equation and one-hot encoding
import numpy as np
from sklearn.linear_model import LinearRegression

# z = z.detach().cpu().numpy()
# y = y.detach().cpu().numpy()
# y_onehot = np.zeros((len(y), 10))
# y_onehot[np.arange(len(y)), y] = 1
# w = np.linalg.solve(z.T @ z, z.T @ y_onehot)
# print(w.shape)
# y_pred = np.argmax(z @ w, axis=1)
# print("accuracy:", np.mean(y == y_pred))

# compute logistic classification using sklearn
# from sklearn.linear_model import LogisticRegression
#
# z = z.detach().cpu().numpy()
# y = y.detach().cpu().numpy()
# model = LogisticRegression(max_iter=1000)
# model.fit(z, y)
# print(model.coef_.shape)
# print("accuracy:", model.score(z, y))

# logistic regression with pytorch
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 10)

    def forward(self, x):
        return self.linear(x)


model = LogisticRegression().to(autoencoder.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

# compute the time elapsed
def opt_it():
    optimizer.zero_grad()
    y_pred = model(z)
    loss = criterion(y_pred, y)
    loss.backward(retain_graph=True)
    optimizer.step()

print(timeit(opt_it, number=100))

# for epoch in range(100):
#
#     print("loss:", loss.item())

print("accuracy:", np.mean(y.cpu().numpy() == np.argmax(y_pred.detach().cpu().numpy(), axis=1)))
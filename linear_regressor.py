
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda")

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Sequential(torch.nn.Linear(4, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 32, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(32, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 1, bias=True))
        self.linear_2 = torch.nn.Sequential(torch.nn.Linear(4, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 32, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(32, 16, bias=True), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 1, bias=True))

    def forward(self, x):
        mask = x[:, -1] == 1

        res = self.linear_1(x[:,:-1]).squeeze(1) * mask + self.linear_2(x[:,:-1]).squeeze(1) * ~mask

       #  if x[-1] == 1:
       #      return self.linear_1(x)
       #  elif x[-1] == -1:
       #      return self.linear_2(x)
       #  else:
       #      raise ValueError("Invalid label")
       # # y_pred = self.linear(x)
        return res

if __name__ == "__main__":
    batch_size = 1000
    dataset = np.loadtxt(open("dataset.txt", "r"), delimiter=",").astype(np.float32)
    x = dataset[:, :-1]
    y = dataset[:, -1]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    #x = torch.tensor(x, dtype=torch.float)
    #y = torch.tensor(y, dtype=torch.float)
    model = Model().to(device)
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pbar = tqdm(range(1000))
    for t in pbar:
        for batch in dataloader:
            batch = batch.to(device)
            y_pred = model(batch[:, :-1])
            loss = criterion(y_pred, batch[:, -1])
            pbar.set_description(f"Loss: {loss.item():.2f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # save checkpoint
        torch.save(model.state_dict(), "model.pt")
        #print(model.linear_1.weight.data)
        #print(model.linear_2.weight.data)

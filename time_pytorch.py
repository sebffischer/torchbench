import torch
import time
import math
from torch import nn
import numpy as np

# 3. Define the timing function
def time_pytorch(epochs, batch_size, n_layers, latent, n, p, device, seed):
    torch.manual_seed(seed)
    # 2. Define a function to create the neural network
    def make_network(p, latent, n_layers):
        layers = [nn.Linear(p, latent), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(latent, latent))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent, 1))
        return nn.Sequential(*layers)

    device = torch.device(device)

    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, 1, device=device)
    Y = X.matmul(beta) + torch.randn(n, 1, device=device) * 0.1

    # Create the network
    net = make_network(p, latent, n_layers)
    net.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    loss_fn = nn.MSELoss()

    def get_batch(step, X, Y, batch_size):
        start_index = step * batch_size
        end_index = min((step + 1) * batch_size, X.size(0))  # Use X.size(0) to get the number of rows
        x_batch = X[start_index:end_index]
        y_batch = Y[start_index:end_index]
        return x_batch, y_batch

    steps = math.ceil(n / batch_size)

    def train_run():
        losses = np.zeros(epochs * steps)   
        for epoch in range(epochs):
            for step in range(steps):
                x, y = get_batch(step, X, Y, batch_size)
                optimizer.zero_grad()
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                losses[epoch * steps + step] = loss.item()
        return losses


    t0 = time.time()
    losses = train_run()
    torch.cuda.synchronize()
    t = time.time() - t0

    return {'time': t, 'losses': losses}


if __name__ == "__main__":
    print(time_pytorch(epochs=10, batch_size=32, n_layers=16, latent=100, n=2000, p=1000, device='cuda', seed=42))
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, InnerProductDecoder

from torch_geometric.datasets import Planetoid
import pickle as pkl
import os
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.ModuleList()
        self.build_encoder(num_features, hidden_size, num_layers)     
        self.num_layers = num_layers  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = x
        for i in range(self.num_layers):
            z = self.encoder[i](z, edge_index)
        return z
    
    def build_encoder(self, num_features, hidden_size, num_layers):
        self.encoder.append(GCNConv(num_features, hidden_size))
        for _ in range(num_layers-1):
            self.encoder.append(GCNConv(hidden_size, hidden_size))
    

if __name__ == "__main__":

    dataname = "Cora"
    data = Planetoid(root="../../data", name=dataname)
    data = data[0]
    print(data)

    n, m, d = data.x.shape[0], data.edge_index.shape[1], data.x.shape[1]
    h = 64
    epochs = 400
    lr = 1e-3
    nlayers = 2

    model = GAE(encoder=Encoder(d, h, nlayers))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        z = model(data)
        loss = model.recon_loss(z, data.edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"EPOCH {epoch+1:05d}/{epochs:05d} {loss:.5f}")

    
    model.eval()
    z = model(data).detach().numpy()
    print(z.shape)

    os.makedirs("outputs", exist_ok=True)
    np.savez(f"outputs/{dataname}", z=z, y=data.y.numpy())
    print("OK")

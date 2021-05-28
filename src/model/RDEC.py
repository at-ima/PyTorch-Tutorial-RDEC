import torch
from torch import nn

from .kmeans import kmeans

class RDEC(nn.Module):
    def __init__(self, encoder, num_clusters, input_data=None, alpha=1):
        super(RDEC, self).__init__()
        
        self.encoder = encoder
        self.alpha = alpha
        
        if input_data is not None:
            self.centroids = nn.Parameter(self.init_centroids(input_data, num_clusters))
        else:
            self.centroids = nn.Parameter(torch.zeros(num_clusters, encoder.emb_dim))
        
    def init_centroids(self, input_data, num_clusters):
        # define initial centroids using k means
        print("initializing centroids by using k-means...")
        x = self.encoder(input_data)
        x = torch.squeeze(x)
        centroids = kmeans(x, num_clusters)
        centroids = centroids.detach().float()
        return centroids
    
    def calc_q_value(self, x):
        # calculation q value using weights of centroids
        norm_squared = torch.sum((x.unsqueeze(1) - self.centroids) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.squeeze(x)
        x = self.calc_q_value(x)
        return x
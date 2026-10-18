import torch.nn as nn
from torchsparse import nn as spnn


# Octant Positional Embedding
class OctantEmbedding(nn.Module):
    def __init__(self, channels=32):
        super(OctantEmbedding, self).__init__()
        self.channels = channels
        self.relative_position_embedding = nn.Embedding(8, channels)

    def forward(self, x):
        coords_abs = x.coords
        coords_rel = coords_abs[:, 1:] % 2
        idxs_rel = coords_rel[:, 0] + coords_rel[:, 1] * 2 + coords_rel[:, 2] * 4
        feats_pos = self.relative_position_embedding(idxs_rel.long()).reshape(
            -1, self.channels
        )

        return feats_pos


#########################################################################################################################################
# Plain Residual Block for Feature Refinement
class SparseResidualBlock(nn.Module):
    def __init__(self, channels=32, k=3):
        super(SparseResidualBlock, self).__init__()

        self.conv0 = spnn.Conv3d(channels, channels, k)
        self.conv1 = spnn.Conv3d(channels, channels, k)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out = self.relu(out + x)
        return out


#########################################################################################################################################
# Occupancy Prediction
class OccupancyPredictor(nn.Module):
    def __init__(self, channels=32):
        super(OccupancyPredictor, self).__init__()

        self.linear0 = nn.Linear(channels, channels)
        self.linear1 = nn.Linear(channels, 16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.linear0(x)
        out = self.relu(out)
        logits = self.linear1(out)
        return logits


#########################################################################################################################################
# Occupancy Context Embedding
class OccupancyEmbedding(nn.Module):
    def __init__(self, channels=32):
        super(OccupancyEmbedding, self).__init__()
        self.channels = channels
        self.emb = nn.Embedding(16, channels)

    def forward(self, prior):
        ctx = self.emb(prior.long()).reshape(-1, self.channels)
        return ctx

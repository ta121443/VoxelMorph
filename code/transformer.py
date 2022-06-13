from types import new_class
import torch
import torch.nn as nn
import torch.nn.functional as nnf

class SpatialTransformer(nn.module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode= mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        def forward(self, src, flow):
            new_locs = self.grid + flow
            shape = flow.shape[2:]

            for i in range(len(shape)): #値の正規化
                new_locs[:, i, ...] = 2 * (new_locs[:,i,...] / (shape[i] - 1) - 0.5)
            
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
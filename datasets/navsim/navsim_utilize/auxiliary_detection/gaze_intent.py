import torch
import torch.nn as nn

class GazeIntent(nn.Module):
    def __init__(
            self,
            in_channles,
            dims,

    ):
        super().__init__()
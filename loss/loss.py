import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
import argparse
import os
from transdiffuser.DDPM.model.dydittraj import TransDiffuserDiT
from ..diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import misc as misc
from util.logger import create_logger
from ..datasets.trajectory_datasets import build_image_dataset
from torch import nn

FLOPS = {
    "DiT-S/2": dict(attn=0.201326592, mlp=0.301989888),
    "DiT-B/2": dict(attn=0.704643072, mlp=1.207959552),
    "DiT-XL/2": dict(attn=1.50994944, mlp=2.717908992)}

class DynamicLoss(nn.Module):
    def __init__(self, 
                token_target_ratio=0.5,
                token_loss_ratio=2., 
                token_minimal=0.1, 
                token_minimal_weight=1.,
                model_name=None,
                model_depth=None
                
                 ):
        super().__init__()
        self.token_target_ratio = token_target_ratio
        self.token_loss_ratio = token_loss_ratio
        self.token_minimal = token_minimal
        self.token_minimal_weight = token_minimal_weight
        
        self.attn_flops = FLOPS[model_name]["attn"]
        self.mlp_flops = FLOPS[model_name]["mlp"]
        self.original_total_flops = (self.attn_flops + self.mlp_flops) * model_depth


    def forward(self, attn_channel_mask, mlp_channel_mask, token_select):
        '''
        head_select: (b, num_layers, num_head)
        '''
        token_loss, real_activate_rate = self._get_token_loss(attn_channel_mask, mlp_channel_mask, token_select)
        
        loss = self.token_loss_ratio * token_loss

        return loss, real_activate_rate
    
    
    def _get_token_loss(self, attn_channel_mask, mlp_channel_mask, token_select):

        mlp_flops = ((mlp_channel_mask.sum(dim=2) / mlp_channel_mask.shape[2]) * (token_select.squeeze(3).sum(dim=2) / token_select.shape[2]) * self.mlp_flops).sum(dim=1)
        
        

        attn_flops = ((attn_channel_mask.sum(dim=2) / attn_channel_mask.shape[2]) * self.attn_flops).sum(dim=1)
        
        total_flops = ((mlp_flops + attn_flops) / self.original_total_flops)
        
        
        
        token_flops_loss = ((total_flops.mean() - self.token_target_ratio)**2)

        if self.token_minimal_weight > 0 :
            token_mean = token_select.mean(-1)
            token_minimal_loss = (self.token_minimal - token_mean).clamp(min=0.).sum()
        else :
            token_minimal_loss = 0

        token_loss = token_flops_loss + self.token_minimal_weight * token_minimal_loss


        return token_loss, total_flops.mean()
    

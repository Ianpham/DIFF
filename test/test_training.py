"""
training

"""

import torch
import torch.nn.functional as F
# backend cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
from glob import glob

import argparse
import os

from transdiffuser.DDPM.model.dydittraj import TransDiffuserDiT 
from DDPM.diffusion import create_diffusion

from diffusers import AutoencoderKL
import misc as misc

from util.logger import create_logger
from .datasets.trajectory_datasets import build_trajectory_dataset

from easydict import EasyDict
from transdiffuser.DDPM.loss.loss import DynamicLoss

import wandb

from utils import clip_grad_norm_

# from util import compute_neuron_head_importance
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay = 0.9999):
    """
    step the EMA model toward the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag = True):
    """
    set requires_grad flag for all parameters in a model
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    end DDP training
    """

    dist.destroy_process_group()
# please consider, I dont think the shape is correct.
def compute_decorrelation_loss(modality_representations):
    """
    Compute decorrelation loss from TransDiffuser paper.
    Encourages orthogonal representations across modalities.
    
    Args:
        modality_representations: Dict of {modality_name: (N, D) tensor}
    
    Returns:
        decorr_loss: Scalar loss value
    """
    if len(modality_representations) < 2:
        return torch.tensor(0.0, device=next(iter(modality_representations.values())).device)
    
    # Stack all modality representations: (N, M, D)
    reps = torch.stack(list(modality_representations.values()), dim=1)
    N, M, D = reps.shape
    
    # Flatten: (N*M, D)
    reps_flat = reps.reshape(-1, D)
    
    # Center features
    reps_centered = reps_flat - reps_flat.mean(dim=0, keepdim=True)
    
    # Compute correlation matrix: (D, D)
    cov = torch.mm(reps_centered.T, reps_centered) / (N * M - 1)
    
    # Decorrelation loss: encourage diagonal correlation matrix
    identity = torch.eye(D, device=cov.device)
    decorr_loss = torch.norm(cov - identity, p='fro') ** 2
    
    return decorr_loss
##################### training loop #######################

def main(args):

    assert torch.cuda.is_available(), "training currretly require at least on GPU"

    misc.init_distributed_mode(args)

    # set up ddp
    # dist.init_process_group("nccl")
    assert args.global_batch_size % misc.get_world_size() == 0, \
        f"Batch size must divisible by world size"
    
    rank = misc.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * misc.get_world_size() + rank
    torch.manual_seed()
    torch.cuda.set_device(device)

    # initialize wandb
    if rank == 0 and args.wandb:
        wandb.init(
            project="transdiffuser",
            name=f'{args.model}_token_ration_{args.token_ration}_decorr_{args.lambda_decoder}'            
        )

    # set up logging
    os.makedirs(args.results_dir, exist_ok = True)
    logger = create_logger(
        output_dir = args.result_dir,
        dist_rank = misc.get_rank(),
        name = f"{args.model}_{int(time())}"
    )

    logger.info(f"Experiement directory created at {args.results_dir}")
    logger.info(f"Staring rank = {rank}, seed = {seed}, world_size = {misc.get_world_size()}.")

    # build trajectory dataset
    dataset = build_trajectory_dataset(args)
    logger.info(f"Dataset length : {len(dataset)}")

    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas= misc.get_world_size(),
        rank= rank,
        shuffle= True,
        seed = args.global_seed
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(args.global_batch_size // misc.get_world_size()),
        shuffle= False,
        sampler=sampler,
        num_workers= args.num_workers,
        pin_memory= True,
        drop_last= True
    )

    # create model
    model = TransDiffuserDiT[args.model](
        traj_dim=args.traj_dim,
        traj_len=args.traj_len,
        ego_dim=args.ego_dim,
        use_modality_specific=args.use_modality_specific,
        parallel=args.parallel,
        modality_config=args.modality_config
    )

    args.model_depth = model.depth

    # load checkpoint if provided
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        checkpoint_model = torch.load(ckpt_path, map_location = lambda storage,loc: storage )

        if args.resume:
            resumed_model = checkpoint_model['model']

        else:
            resumed_model = checkpoint_model

        msg = model.load_state_dict(resumed_model, strict = False)
        logger.info(f"loader checkpoint: {msg}")

    n_parameters = sum(p.numel() for p in model.parameters if p.requires_grad)
    logger.info(f'Number of trainable params (M): {n_parameters / 1e6::2f}')

    model_without_ddp = model

    # create EMA model
    ema = deepcopy(model_without_ddp).to(device)
    requires_grad(ema, False)

    # set up DDP
    if args.distributed:
        model = DDP(model_without_ddp.to(device), device_ids=[rank])
    
    else:
        model = model_without_ddp.to(device)
    
    # create diffusion (DDPM)
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"TransDiffuser Parameters: {sum(p.numel() for p in model.parameters()):, }")

    # setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=0)
    logger.info(f"Learning rate set to {args.lr}")

    if args.resume:
        logger.info(f'Resuming EMA from {ckpt_path}')
        resumed_ema = checkpoint_model['ema']
        ema.load_state_dict(resumed_ema)

    model.train()
    ema.eval()

    # set up dynamics loss 
    from easydict import EasyDict
    select_config = EasyDict(
        token_ratio = 2.,
        token_target_ratio = args.token_ratio,
        token_minimal = 0.,
        token_minimal_weight = 0.,
    )

    token_loss_func = DynamicLoss(
        token_target_ratio= select_config.token_target_ratio,
        token_loss_ratio= select_config.token_ratio,
        token_minimal=select_config.token_minimal,
        token_minimal_weight = select_config.token_minimal_weight,
        model_name = args.model,
        model_depth= args.model_depth
    )


    # training variable
    train_steps = 0
    if args.resume:
        train_steps = int(args.ckpt.split("/")[-1].split(".")[0])
    
    log_steps = 0
    running_loss = 0
    running_dynamic_loss = 0
    running_diffusion_loss = 0
    running_decorr_loss = 0
    start_time = time()
    flag = False


    # for timestep sampling
    if args.t_sample:
        t_sampling_choice = [0, 250, 500, 750, 1000]

        range_choices = torch.arange(
            int(args.global_batch_size // misc.get_world_size()),
            device = device
        ) % 4

        lower_bound = torch.tensor(t_sampling_choice[:-1], device = device)[range_choices]

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch=epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for batch in loader:
            # 1. prepare multi-model context 
            context = {}

            # add each modality to context dictionary
            for modality_name in args.modality_names:
                if modality_name in batch:
                    context[modality_name] = batch[modality_name].to(device)

        

            # add ego state
            if 'ego_state' in batch:
                context['ego'] = batch['ego_state'].to(device)

            # add action history
            if 'action_history' in batch:
                context['action_history'] = batch['action_history'].to(device)

            #groud truth
            clean_trajectory = batch['future_trajectory'].to(device)

            # hsape [N, traj_len, traj_dim]


            # sample diffusion step
            if args.t_sample:
                t = torch.randint(0, 250, (clean_trajectory.shape[0],), device= device) + lower_bound

            else:
                t = torch.randint(0, diffusion.num_timesteps, (clean_trajectory.shape[0],), device = device)


            # forward diffusion: add noise to trajectory
            noise = torch.randn_like(clean_trajectory)

            noisy_trajectory = diffusion.q_sample(clean_trajectory, t, noise)

            # flop target ratio
            if args.warmup and (train_steps <= args.warmup_step):
                flops_target_ratio = 1.0 - (1.0 - args.token_ratio) * train_steps
            
            else:
                flops_target_ratio = args.token_ratio

            if train_steps < 100:
                flops_target_ratio = 1.0

            token_loss_func.token_loss_ratio = flops_target_ratio


            # forwards pass:predict noise
            complete_model = (train_steps% 100 == 0)

            predicted_noise, modality_representations, attn_masks, mlp_masks, token_masks = model(
                context = context,
                noisy_trajectory = noisy_trajectory,
                t = t,
                complete_model = complete_model
            )

            # loss
            # diffusion loss
            diffusion_loss = F.mse_loss(predicted_noise, noise)
            # dynamic loss : token selection efficiency
            dynamic_loss, real_activate_rate = token_loss_func(
                attn_masks, 
                mlp_masks,
                token_masks
            )

            # decorralation loss 
            decorr_loss = compute_decorrelation_loss(modality_representations)


            # total loss
            total_loss = (diffusion_loss + args.lambda_dynamic * dynamic_loss + args.lambda_decoor * decorr_loss)


            # backward and optimize
            opt.zero_grad()
            total_loss.backward()

            # gradient clipping
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                max_norm=args.clip_max_norm,
                clip_grad = True
            )

            if gradient_norm < args.clip_grad_norm:
                opt.step()

            else:
                logger.info(
                    f"step {train_steps}: skipping update due to large gradient norm {gradient_norm:.2f}"
                )

            # update EMA
            update_ema(ema, model.module if args.distrubuted else model)

            # logging
            running_loss += total_loss.item()
            running_dynamic_loss += dynamic_loss.item()
            running_diffusion_loss += diffusion_loss.item()
            running_decorr_loss += decorr_loss.item()  # NEW
            
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce loss history over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_dynamic_loss = torch.tensor(running_dynamic_loss / log_steps, device=device)
                avg_diffusion_loss = torch.tensor(running_diffusion_loss / log_steps, device=device)
                avg_decorr_loss = torch.tensor(running_decorr_loss / log_steps, device=device)
                
                avg_loss = misc.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_dynamic_loss = misc.all_reduce(avg_dynamic_loss, op=dist.ReduceOp.SUM)
                avg_diffusion_loss = misc.all_reduce(avg_diffusion_loss, op=dist.ReduceOp.SUM)
                avg_decorr_loss = misc.all_reduce(avg_decorr_loss, op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / misc.get_world_size()
                avg_dynamic_loss = avg_dynamic_loss.item() / misc.get_world_size()
                avg_diffusion_loss = avg_diffusion_loss.item() / misc.get_world_size()
                avg_decorr_loss = avg_decorr_loss.item() / misc.get_world_size()
                
                # Log to wandb
                if rank == 0 and args.wandb:
                    wandb.log({
                        "Train Loss": avg_loss,
                        "Diffusion Loss": avg_diffusion_loss,
                        "Dynamic Loss": avg_dynamic_loss,
                        "Decorrelation Loss": avg_decorr_loss,  # NEW
                        "FLOPS Ratio": flops_target_ratio,
                        "Activate Ratio": real_activate_rate.item()
                    })
                
                # Log to console
                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Total Loss: {avg_loss:.4f} "
                    f"Diffusion Loss: {avg_diffusion_loss:.4f} "
                    f"Decorrelation Loss: {avg_decorr_loss:.4f} "  # NEW
                    f"Dynamic Loss: {avg_dynamic_loss:.4f} "
                    f"FLOPS Ratio: {flops_target_ratio:.4f} "
                    f"Real Activate Ratio: {real_activate_rate.item():.4f} "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                
                # Reset monitoring variables
                running_loss = 0
                running_diffusion_loss = 0
                running_decorr_loss = 0
                running_dynamic_loss = 0
                log_steps = 0
                start_time = time()
            
            # save checkpoint
            if (train_steps == args.total_train_steps) or \
               (train_steps % args.ckpt_every == 0 and train_steps > 0):
                if rank == 0:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{args.results_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps == args.total_train_steps:
                flag = True
                break
        
        if flag:
            break
    
    model.eval()
    if rank == 0 and args.wandb:
        wandb.finish()
    
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to trajectory dataset")
    parser.add_argument("--results-dir", type=str, default="results")
    
    # Model
    parser.add_argument("--model", type=str, default="TransDiffuser-Base",
                        help="Model name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to checkpoint")
    parser.add_argument("--resume", action='store_true', default=False,
                        help="Resume from checkpoint")
    
    # Trajectory parameters (NEW)
    parser.add_argument("--traj-dim", type=int, default=5,
                        help="Trajectory dimension (x, y, vel_x, vel_y, heading)")
    parser.add_argument("--traj-len", type=int, default=10,
                        help="Number of future waypoints to predict")
    parser.add_argument("--ego-dim", type=int, default=8,
                        help="Ego state dimension")
    
    # Multi-modal settings (NEW)
    parser.add_argument("--use-modality-specific", action='store_true', default=True,
                        help="Use modality-specific encoders")
    parser.add_argument("--parallel", action='store_true', default=True,
                        help="Use parallel fusion (vs hierarchical)")
    parser.add_argument("--modality-names", nargs='+', 
                        default=['lidar', 'img', 'BEV'],
                        help="List of modality names to use")
    parser.add_argument("--modality-config", type=str, default=None,
                        help="Path to modality config JSON")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Loss weights (NEW)
    parser.add_argument("--lambda-decorr", type=float, default=0.1,
                        help="Weight for decorrelation loss")
    parser.add_argument("--lambda-dynamic", type=float, default=1.0,
                        help="Weight for dynamic token loss")
    
    # Dynamic token selection
    parser.add_argument("--token-ratio", type=float, default=0.5)
    parser.add_argument("--warmup", action='store_true', default=False)
    parser.add_argument("--warmup-step", type=int, default=1000)
    
    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--wandb", action='store_true', default=False)
    
    # Training settings
    parser.add_argument("--total-train-steps", type=int, default=150000)
    parser.add_argument("--clip-max-norm", type=float, default=15.0)
    parser.add_argument("--t-sample", action='store_true', default=False)
    
    # Distributed
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    
    args = parser.parse_args()
    main(args)


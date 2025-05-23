import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Add this import
import lpips
from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import get_solvers, move_tensor_to_device, parse_arguments, set_seed_everything

from dataset import load_data_from_dir, LTTDataset
from latent_to_timestep_model import LTT_model
from models import prepare_stuff
import torch.optim.lr_scheduler as lr_scheduler


args = parse_arguments()
set_seed_everything(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dataset
data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'

# Initialize TensorBoard writer
learning_rate = args.lr_time_1
run_name = f"steps_{args.steps}_model_lr{learning_rate}_batch{args.main_train_batch_size}_{args.log_suffix}"
log_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_zeroshot_timesteps/{run_name}"
model_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_zeroshot_timesteps/models"
model_path = os.path.join(model_dir, run_name)
writer = SummaryWriter(log_dir)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

def custom_collate_fn(batch):
    collated_batch = []
    for samples in zip(*batch):
        if any(item is None for item in samples):
            collated_batch.append(None)
        else:
            collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    return collated_batch
# Initialize diffusion model components

wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)
solver, steps, solver_extra_params = get_solvers(
    args.solver_name,
    NFEs=args.steps,
    order=args.order,
    noise_schedule=noise_schedule,
    unipc_variant=args.unipc_variant,
)

valid_dataset = LTTDataset(dir=os.path.join(data_dir, "validation"), size=args.num_valid, train_flag=False, use_optimal_params=False)
train_dataset = LTTDataset(dir=os.path.join(data_dir, "train"), size=args.num_train, train_flag=True, use_optimal_params=False)


model = LTT_model(steps = steps, mlp_dropout=args.mlp_dropout)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
step_size = args.training_rounds_v1 * len(train_dataset) // args.main_train_batch_size // 100 #we decrease 100 times which roughly equals a decrease of 99% of learning rate in the end
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.95)  # Decrease LR by a factor of 0.1 every 1 epochs



wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)
solver, steps, solver_extra_params = get_solvers(
    args.solver_name,
    NFEs=args.steps,
    order=args.order,
    noise_schedule=noise_schedule,
    unipc_variant=args.unipc_variant,
)

training_config = TrainingConfig(
    train_data=train_dataset,
    valid_data=valid_dataset,
    train_batch_size=args.main_train_batch_size,
    valid_batch_size=args.main_valid_batch_size,
    lr_time_1=args.lr_time_1,
    shift_lr=args.shift_lr,
    shift_lr_decay=args.shift_lr_decay,
    min_lr_time_1=args.min_lr_time_1,
    win_rate=args.win_rate,
    patient=args.patient,
    lr_time_decay=args.lr_time_decay,
    momentum_time_1=args.momentum_time_1,
    weight_decay_time_1=args.weight_decay_time_1,
    loss_type=args.loss_type,
    visualize=args.visualize,
    no_v1=args.no_v1,
    prior_timesteps=args.gits_ts,
    match_prior=args.match_prior,
)
model_config = ModelConfig(
    net=wrapped_model,
    decoding_fn=decoding_fn,
    noise_schedule=noise_schedule,
    solver=solver,
    solver_name=args.solver_name,
    order=args.order,
    steps=steps,
    prior_bound=args.prior_bound,
    resolution=latent_resolution,
    channels=latent_channel,
    time_mode=args.time_mode,
    solver_extra_params=solver_extra_params,
    device=device,
)
trainer = LD3Trainer(model_config, training_config)


dis_model = DiscretizeModelWrapper( #Changed through LTT
        lambda_max=trainer.lambda_max,
        lambda_min=trainer.lambda_min,
        noise_schedule=trainer.noise_schedule,
        time_mode = trainer.time_mode,
    )


best_loss = 1
# Forward pass
for i in range(args.training_rounds_v1):
    print(f"\n epoch: {i}")
    for iter, batch in enumerate(trainer.train_loader):
        img, latent, _ = batch

        img = img.to(device)
        latent = latent.to(device)
        
        outputs = model(latent)

        timesteps_list = dis_model.convert(outputs)
        x_next_list = noise_schedule.prior_transformation(latent)

        x_next_computed = []
        for timestep, x_next in zip(timesteps_list, x_next_list):
            x_next = solver.sample_simple(
                model_fn=trainer.net,
                x=x_next.unsqueeze(0),
                timesteps=timestep,
                order=args.order,
                NFEs=steps,
                condition=None,
                unconditional_condition=None,
                **solver_extra_params,
            )
            x_next_computed.append(x_next)

        x_next_computed = torch.cat(x_next_computed, dim=0)          
        loss_vector = trainer.loss_fn(img.float(), x_next_computed.float()).squeeze()
        loss = loss_vector.mean()

        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        loss.backward()
        writer.add_scalar(f"Train/Loss", loss.item(), i*len(trainer.train_loader)+iter) 
        optimizer.step() #does this ever change?
        optimizer.zero_grad()
        scheduler.step()

        if iter % 500 == 0:
            
            with torch.no_grad():
                model.eval()
                for batch in trainer.valid_only_loader:
                    img, latent, _ = batch

                    img = img.to(device)
                    latent = latent.to(device)
                    
                    outputs = model(latent)

                    timesteps_list = dis_model.convert(outputs)
                    x_next_list = noise_schedule.prior_transformation(latent)

                    x_next_computed = []
                    for timestep, x_next in zip(timesteps_list, x_next_list):
                        x_next = solver.sample_simple(
                            model_fn=trainer.net,
                            x=x_next.unsqueeze(0),
                            timesteps=timestep,
                            order=args.order,
                            NFEs=steps,
                            condition=None,
                            unconditional_condition=None,
                            **solver_extra_params,
                        )
                        x_next_computed.append(x_next)

                    x_next_computed = torch.cat(x_next_computed, dim=0)          
                    loss_vector = trainer.loss_fn(img.float(), x_next_computed.float()).squeeze()
                    loss = loss_vector.mean()

                    writer.add_scalar(f"Valid/Loss", loss.item(), i*len(trainer.train_loader)+iter) 
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Validated on iter {i*len(trainer.train_loader)+iter}: Loss = {loss.item()}, Learning Rate = {current_lr}")
                
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), model_path)
                model.train()

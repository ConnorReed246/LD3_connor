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
from latent_to_timestep_model import LTT_model, Delta_LTT_model
from models import prepare_stuff
import torch.optim.lr_scheduler as lr_scheduler
from utils import visual


args = parse_arguments()
set_seed_everything(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dataset
data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'
model_dir = "runs/RandomModels"
steps = 5
optimal_params_path = args.data_dir #opt_t_clever_initialisation

# Initialize TensorBoard writer
learning_rate = args.lr_time_1
run_name = f"model_lr{learning_rate}_batch{args.main_train_batch_size}_nTrain{args.num_train}_{args.log_suffix}"
log_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_delta_timesteps/{run_name}"
model_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_delta_timesteps/models"
writer = SummaryWriter(log_dir)
process_img_dir = os.path.join("/netpool/homes/connor/DiffusionModels/LD3_connor/runs_delta_timesteps/process_images/", run_name)
os.makedirs(process_img_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

# Initialize diffusion model components

wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)
solver, steps, solver_extra_params = get_solvers(
    args.solver_name,
    NFEs=args.steps,
    order=args.order,
    noise_schedule=noise_schedule,
    unipc_variant=args.unipc_variant,
)
order = args.order  

def custom_collate_fn(batch):
    collated_batch = []
    for samples in zip(*batch):
        if any(item is None for item in samples):
            collated_batch.append(None)
        else:
            collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    return collated_batch

valid_dataset = LTTDataset(dir=os.path.join(data_dir, "validation"), size=args.num_valid, train_flag=False, use_optimal_params=False,optimal_params_path=optimal_params_path) 
train_dataset = LTTDataset(dir=os.path.join(data_dir, "train"), size=args.num_train, train_flag=True, use_optimal_params=False, optimal_params_path=optimal_params_path)

delta_ltt_model = Delta_LTT_model(steps = steps, mlp_dropout=args.mlp_dropout)
delta_ltt_model = delta_ltt_model.to(device)
optimizer = torch.optim.AdamW(delta_ltt_model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Decrease LR by a factor of 0.1 every 1 epochs



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

for i in range(args.training_rounds_v1):
    for iter, batch in enumerate(trainer.train_loader):
        img, latent, _ = batch
        img = img.to(device)
        latent = latent.to(device)

        x_next_list = trainer.noise_schedule.prior_transformation(latent) #Multiply with timestep in edm case (x80 in beginning)

        x_next_computed = []
        x_next_list_computed = []
        for x in x_next_list:
            x_next, x_list = trainer.solver.delta_sample_simple(
                model_fn=trainer.net,
                delta_ltt=delta_ltt_model,
                x=x.unsqueeze(0),
                order=trainer.order,
                steps = trainer.steps,
                start_timestep = 80,
                NFEs=trainer.steps,
                condition=None,
                unconditional_condition=None,
                **trainer.solver_extra_params,
            )
            x_next_computed.append(x_next)#This was wrong the whole time?
            x_next_list_computed.append(x_list)
        
        x_next_computed = torch.cat(x_next_computed, dim=0) 
        loss_vector = trainer.loss_fn(img.float(), x_next_computed.float()).squeeze()
        loss = loss_vector.mean()
        torch.nn.utils.clip_grad_norm_(delta_ltt_model.parameters(), 5.0)
        loss.backward()
        writer.add_scalar(f"Train/Loss", loss.item(), i*len(trainer.train_loader)+iter) 
        optimizer.step() #does this ever change?
        optimizer.zero_grad()

        if iter % 500 == 0:
            
            visual(torch.cat(x_list),  f"{process_img_dir}/train_{i*len(trainer.train_loader)+iter}.png") 
            with torch.no_grad():
                delta_ltt_model.eval()
                for batch in trainer.valid_only_loader:
                    img, latent, _ = batch
                    latent = latent.to(device)
                    img = img.to(device)

                    x_next_list = trainer.noise_schedule.prior_transformation(latent) #Multiply with timestep in edm case (x80 in beginning)
                    x_next_computed = []
                    x_next_list_computed = []
                    for x in x_next_list:
                        x_next, x_list = trainer.solver.delta_sample_simple(
                            model_fn=trainer.net,
                            delta_ltt=delta_ltt_model,
                            x=x.unsqueeze(0),
                            order=trainer.order,
                            steps = trainer.steps,
                            start_timestep = 80,
                            NFEs=trainer.steps,
                            condition=None,
                            unconditional_condition=None,
                            **trainer.solver_extra_params,
                        )
                        x_next_computed.append(x_next)#This was wrong the whole time?
                    
                    x_next_computed = torch.cat(x_next_computed, dim=0) 
                    loss_vector = trainer.loss_fn(img.float(), x_next_computed.float()).squeeze()
                    loss = loss_vector.mean()
                    visual(x_next_computed, f"{process_img_dir}/valid_{i*len(trainer.train_loader)+iter}.png")
                    writer.add_scalar(f"Valid/Loss", loss.item(), i*len(train_dataset)+iter) 
                    print(f"Validated on iter{i*len(trainer.train_loader)+iter}: {loss.item()}")
                delta_ltt_model.train()
    scheduler.step()


# Save the model
model_path = os.path.join(model_dir, run_name)
torch.save(delta_ltt_model.state_dict(), model_path)









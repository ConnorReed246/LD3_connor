import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from dataset import load_data_from_dir, LTTDataset
from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import (
    get_solvers,
    parse_arguments,
    adjust_hyper,
    set_seed_everything,
    move_tensor_to_device,
    save_rng_state,
    visual
)
from models import prepare_stuff
from latent_to_timestep_model import Delta_LTT_model, Delta_LTT_model_using_Bottleneck
import lpips



start_time = time.time()
args = parse_arguments()

print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
set_seed_everything(args.seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dataset
data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'
model_dir = "runs/RandomModels"
steps = 5

learning_rate = args.lr_time_1
run_name = f"model_lr{learning_rate}_batch{args.main_train_batch_size}_nTrain{args.num_train}_{args.log_suffix}"
log_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/{run_name}"
model_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/models"
model_path = os.path.join(model_dir, run_name)
writer = SummaryWriter(log_dir)
os.makedirs(model_dir, exist_ok=True)

def custom_collate_fn(batch):
    collated_batch = []
    for samples in zip(*batch):
        if any(item is None for item in samples):
            collated_batch.append(None)
        else:
            collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    return collated_batch

valid_dataset = LTTDataset(dir=os.path.join(data_dir, "validation"), size=args.num_valid, train_flag=False, use_optimal_params=False) 
train_dataset = LTTDataset(dir=os.path.join(data_dir, "train"), size=args.num_train, train_flag=True, use_optimal_params=False)

wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)
adjust_hyper(args, latent_resolution, latent_channel)
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


params = torch.nn.Parameter(torch.ones(args.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)
# params = torch.nn.Parameter(torch.tensor([0.1140, 0.1652, 0.1298, 0.1056, 0.1084, 0.3770], dtype=torch.float32).cuda(), requires_grad=True)
optimizer = torch.optim.RMSprop(
    [params], 
    lr=training_config.lr_time_1,
    momentum=training_config.momentum_time_1,
    weight_decay=training_config.weight_decay_time_1,
)

step_size = args.training_rounds_v1 * len(train_dataset) // args.main_train_batch_size // 100 #we decrease 100 times which roughly equals a decrease of 99% of learning rate in the end
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.95)
best_loss = 1

for i in range(args.training_rounds_v1):
    for iter, batch in enumerate(trainer.train_loader):
        img, latent, _ = batch
        img, latent = move_tensor_to_device(img, latent, device=device)
        params_softmax = F.softmax(params, dim=0)
        timestep = dis_model.convert(params_softmax.unsqueeze(0))
        x_next = trainer.noise_schedule.prior_transformation(latent)
        x_next = trainer.solver.sample_simple(
            model_fn=trainer.net,
            x=x_next,
            timesteps=timestep[0],
            order=trainer.order,
            NFEs=trainer.steps,
            **trainer.solver_extra_params,
                )
        x_next = trainer.decoding_fn(x_next)
        trainer.loss_vector = trainer.loss_fn(img.float(), x_next.float()).squeeze()
        loss = trainer.loss_vector.mean() 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        writer.add_scalar(f"Train/Loss", loss.item(), i*len(trainer.train_loader)+iter) 


        if iter % (step_size // 3) == 0:
            with torch.no_grad():
                for batch in trainer.valid_only_loader:
                    img, latent, _ = batch
                    img, latent = move_tensor_to_device(img, latent, device=device)

                    params_softmax = F.softmax(params, dim=0)
                    timestep = dis_model.convert(params_softmax.unsqueeze(0))
                    x_next = trainer.noise_schedule.prior_transformation(latent)
                    x_next = trainer.solver.sample_simple(
                        model_fn=trainer.net,
                        x=x_next,
                        timesteps=timestep[0],
                        order=trainer.order,
                        NFEs=trainer.steps,
                        **trainer.solver_extra_params,
                            )
                    x_next = trainer.decoding_fn(x_next)
                    trainer.loss_vector = trainer.loss_fn(img.float(), x_next.float()).squeeze()
                    loss = trainer.loss_vector.mean() 
                    writer.add_scalar(f"Valid/Loss", loss.item(), i*len(train_dataset)+iter) 
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Validated on iter {i*len(trainer.train_loader)+iter}: Loss = {loss.item()}, Learning Rate = {current_lr}") 
                    print("params_softmax: ", params_softmax)
                    print("timestep: ", timestep)
                    if loss < best_loss:
                        best_loss = loss
                        torch.save({
                            'params_softmax': params_softmax.cpu().detach(),
                            'timesteps': timestep.cpu().detach(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, model_path)


print("Time taken: ", time.time() - start_time)








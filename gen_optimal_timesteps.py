import torch
from torch.nn import functional as F
import time
import os

from dataset import load_data_from_dir
from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import (
    get_solvers,
    parse_arguments,
    adjust_hyper,
    set_seed_everything,
    move_tensor_to_device
)
from models import prepare_stuff


def gen_optimal_timesteps(args):
    start_time = time.time()
    set_seed_everything(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)
    adjust_hyper(args, latent_resolution, latent_channel)
    solver, steps, solver_extra_params = get_solvers(
        args.solver_name,
        NFEs=args.steps,
        order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=args.unipc_variant,
    )
    latents, targets, _, _, _ = load_data_from_dir( #this is what we take from trainig, targets are original images and latens latent goal
        data_folder=args.data_dir, limit=args.num_train + args.num_valid, use_optimal_params=False
    )
    
    training_config = TrainingConfig(
        train_data=latents,
        valid_data=latents,
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


    # loss_matrix = torch.zeros(len(targets), args.training_rounds_v1)
    # grad_matrix = torch.zeros(len(targets), args.training_rounds_v1, args.steps + 1)

    for i, (img, latent) in enumerate(zip(targets, latents)):
        
        loss_list = torch.zeros(args.n_trials)
        params_softmax_list = torch.zeros(args.n_trials, args.steps + 1)
        img, latent = move_tensor_to_device(img, latent, device = device)
        for r in range(args.n_trials):
            params = torch.nn.Parameter(torch.ones(args.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)
            if r > 0:
                params.data += torch.rand(params.size()).cuda() - 0.5
            
            optimizer = torch.optim.RMSprop(
                [params], 
                lr=training_config.lr_time_1,
                momentum=training_config.momentum_time_1,
                weight_decay=training_config.weight_decay_time_1,
            )

            for j in range(args.training_rounds_v1):
                params_softmax = F.softmax(params, dim=0)
                timestep = dis_model.convert(params_softmax)
                x_next = trainer.noise_schedule.prior_transformation(latent)
                x_next = trainer.solver.sample_simple(
                    model_fn=trainer.net,
                    x=x_next,
                    timesteps=timestep,
                    order=trainer.order,
                    NFEs=trainer.steps,
                    **trainer.solver_extra_params,
                    )
                x_next = trainer.decoding_fn(x_next)
                trainer.loss_vector = trainer.loss_fn(img.float(), x_next.float()).squeeze()
                loss = trainer.loss_vector.mean() 
                loss.backward()
                # grad_matrix[i,j] = optimizer.param_groups[0]["params"][0].grad
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                # loss_matrix[i, j] = loss.item()
            loss_list[r] = loss
            params_softmax_list[r] = params_softmax

        #sort both lists
        loss_list, indices = torch.sort(loss_list)
        params_softmax_list = params_softmax_list[indices]
        print("img: ", i)
        print("loss_list: ", loss_list)
        print("params_softmax_list: \n", params_softmax_list)
        torch.save((params_softmax_list, loss_list), os.path.join(args.data_dir, f'optimal_params_{i:06d}_N{args.n_trials}_steps{args.steps}.pth'))
        print("-------------------------------")


            # torch.save(timestep, os.path.join(args.data_dir, f"optimal_timestep_{i}.pt"))
    
    # torch.save(loss_matrix, os.path.join(args.data_dir, f"loss_matrix.pt"))
    # torch.save(grad_matrix, os.path.join(args.data_dir, f"loss_grad_matrix.pt"))
    print("Time taken: ", time.time() - start_time)


if __name__ == "__main__":

    """relevant parameters:
    "--all_config", "configs/cifar10.yml",
    "--data_dir", "train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0",
    "--num_train", "20",
    "--num_valid", "10",
    "--steps", "10",
    training_rounds_v1

    """
    args = parse_arguments()
    gen_optimal_timesteps(args)
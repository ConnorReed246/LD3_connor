import torch
from torch.nn import functional as F
import time
import os

from dataset import load_data_from_dir, LTTDataset
from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import (
    get_solvers,
    parse_arguments,
    adjust_hyper,
    set_seed_everything,
    move_tensor_to_device,
    save_rng_state
)
from models import prepare_stuff
import matplotlib.pyplot as plt


def gen_optimal_timesteps(args):
    start_time = time.time()
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    set_seed_everything(args.seed)

    train_or_validation = args.train_or_validation
    train_bool = True if train_or_validation == "train" else False
    dir = os.path.join(args.data_dir, train_or_validation)
    dataset = LTTDataset(os.path.join(dir), train_flag= train_bool, size=500000)

    rng_path = os.path.join(dir, "opt_t_rng_states") #_clever_initialisation
    os.makedirs(rng_path, exist_ok=True)
    opt_t_path = os.path.join(dir, "opt_t") #_clever_initialisation
    os.makedirs(opt_t_path, exist_ok=True)

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
    
    training_config = TrainingConfig(
        train_data=dataset,
        valid_data=dataset,
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


    rng_files = [f for f in os.listdir(rng_path) if f.startswith("rng_state_") and f.endswith(".pt")]
    if rng_files:
        latest_rng_file = max(rng_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        count = int(latest_rng_file.split('_')[-1].split('.')[0])
        latest_rng_file_path = os.path.join(rng_path, latest_rng_file)
        print(f"Loaded RNG state from {latest_rng_file_path}")
    else:
        print("No RNG state files found, starting from scratch.")
        count = 0

    while True:
        if count >= args.total_samples:
            break
        
        loss_list = torch.zeros(args.n_trials)
        params_softmax_list = torch.zeros(args.n_trials, args.steps + 1)
        
        try: 
            img, latent, _ = dataset[count]
        except IndexError:
            print(f"Reached end of generated images at count {count}")
            break

        img, latent = move_tensor_to_device(img, latent, device=device)
        for r in range(args.n_trials):
            params = torch.nn.Parameter(torch.ones(args.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)
            # params = torch.nn.Parameter(torch.tensor([0.1140, 0.1652, 0.1298, 0.1056, 0.1084, 0.3770], dtype=torch.float32).cuda(), requires_grad=True)

            if r > 0:
                params.data += (torch.rand(params.size()).cuda() - 0.5) #* 0.1
            
            optimizer = torch.optim.RMSprop(
                [params], 
                lr=training_config.lr_time_1,
                momentum=training_config.momentum_time_1,
                weight_decay=training_config.weight_decay_time_1,
            )

            for j in range(args.training_rounds_v1):
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
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            loss_list[r] = loss
            params_softmax_list[r] = params_softmax



        #sort both lists
        loss_list, indices = torch.sort(loss_list)
        params_softmax_list = params_softmax_list[indices]
        # print("img: ", count)
        # print("loss_list: ", loss_list)
        # print("params_softmax_list: \n", params_softmax_list)
        torch.save((params_softmax_list, loss_list), os.path.join(opt_t_path, f'optimal_params_{count:06d}_N{args.n_trials}_steps{args.steps}.pth'))
        # print("-------------------------------")
        
        count += 1
        if count % 100 == 0:
            save_rng_state(os.path.join(rng_path, f"rng_state_{count:06d}"))
            print(f"Saved RNG states at count {count}")


    print("Time taken: ", time.time() - start_time)




    # loss_matrix = torch.zeros(args.num_train + args.num_valid, args.training_rounds_v1)
    # plt.figure(figsize=(10, 6))
    # for i in range(loss_matrix.size(0)):
    #     plt.plot(loss_matrix[i].cpu().numpy(), label=f'Image {i}')
    # plt.legend()
    # plt.title('LPIPS loss per image')
    # plt.xlabel('Iteration')
    # plt.ylabel('LPIPS Loss')
    # plt.savefig(os.path.join(args.data_dir, 'OptimalTimesteps/loss_matrix.png'))


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
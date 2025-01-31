import torch

import logging
import os
import sys
import time

from dataset import load_data_from_dir, LD3Dataset
from trainer import LD3Trainer, ModelConfig, TrainingConfig
from utils import (
    create_desc,
    is_trained,
    get_solvers,
    parse_arguments,
    adjust_hyper,
    save_arguments_to_yaml,
    set_seed_everything,
    Tensorboard_Logger
)
from models import prepare_stuff


def setup_logging(log_dir):
    """
    checked!
    """
    # Reset logging configuration
    logging.shutdown()
    import importlib
    importlib.reload(logging)
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return logging


def main(args):
    start = time.time()
    if args.use_ema:
        print("Auto update use_ema to False for training")
        args.use_ema = False 

    set_seed_everything(args.seed)
    #load model and set grad to false, wrap as model that predicts noise, ingnore net, image_resolution, and image_channels 
    wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)

    adjust_hyper(args, latent_resolution, latent_channel)
    desc = create_desc(args)

    log_dir = os.path.join(args.log_path, desc)

    if is_trained(log_dir):

        if args.force_train:
            print("Retraining and removing old logs")
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
        else:
            print("Skip training")
            return
    else:
        print("The model hasn't been trained yet. Perform training")
    os.makedirs(log_dir, exist_ok=True)
    save_arguments_to_yaml(args, os.path.join(log_dir, "config.yml"))
    setup_logging(log_dir)

    if args.use_tensorboard:
        writer = Tensorboard_Logger.get_writer(log_dir = "runs/" + desc + "_" + time.strftime("%m-%d_%H:%M", time.localtime(start)))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models -> used to increase sampling speed
    solver, steps, solver_extra_params = get_solvers(
        args.solver_name,
        NFEs=args.steps,
        order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=args.unipc_variant,
    )
    latents, targets, conditions, unconditions = load_data_from_dir( #this is what we take from trainig, targets are original images and latens latent goal
        data_folder=args.data_dir, limit=args.num_train + args.num_valid
    )

    valid_dataset = LD3Dataset(
        latents[: args.num_valid],
        targets[: args.num_valid],
        conditions[: args.num_valid],
        unconditions[: args.num_valid],
    )
    if args.num_valid > 0 :
        train_dataset = LD3Dataset(
            latents[args.num_valid :],
            targets[args.num_valid :],
            conditions[args.num_valid :],
            unconditions[args.num_valid :],
        )
    else:
        valid_dataset = train_dataset

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
        snapshot_path=log_dir,
        device=device,
    )
    trainer = LD3Trainer(model_config, training_config)
        
    trainer.train(args.training_rounds_v1)
    end = time.time()
    logging.info(f"Training time: {end - start}")
    Tensorboard_Logger.close()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

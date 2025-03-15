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
# from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import get_solvers, move_tensor_to_device, parse_arguments, set_seed_everything

from dataset import load_data_from_dir, LTTDataset

# Fully connected neural network with one hidden layer
class LTT_model(nn.Module):
    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0):
        super().__init__()

        self.unet = SimpleUNet_Encoding(
            in_channels=3
        )
        self.mlp = SimpleMLP(
            input_size=256 * 8 * 8,
            output_size=steps + 1,
            hidden_size=100,
            dropout=mlp_dropout
        )
    
    def forward(self, x):
        out = self.unet(x)
        out = self.mlp(out)

        return out

#https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
class SimpleUNet_Encoding(torch.nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x):
            residual = self.shortcut(x)
            x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
            x = self.bn2(self.conv2(x))
            return F.leaky_relu(x + residual, 0.1)
    class DownSample(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = SimpleUNet_Encoding.DoubleConv(in_channels, out_channels)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            down = self.conv(x)
            p = self.pool(down)

            return down, p

    def __init__(self, in_channels):
        super().__init__()
        self.down_convolution_1 = SimpleUNet_Encoding.DownSample(in_channels, 64)
        self.down_convolution_2 = SimpleUNet_Encoding.DownSample(64, 128)
        self.bottle_neck = SimpleUNet_Encoding.DoubleConv(128, 256)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)

        b = self.bottle_neck(p2)
        return b



class SimpleMLP(nn.Module):
    class L1NormLayer(nn.Module):
        def forward(self, x):
            return x / x.abs().sum(dim=1, keepdim=True)
    

    def __init__(self, input_size, output_size, hidden_size=100, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.l1_norm = SimpleMLP.L1NormLayer()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Add dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Add dropout
        x = self.fc3(x)
        x = F.softplus(x)
        # x = torch.sigmoid(x) 
        x = self.l1_norm(x)
        
        
        # x = torch.softmax(x, dim=1)  # Apply softmax along the class dimension

        return x


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dataset
    data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'
    steps = 5
    num_train = 160000 #40000 if clever
    num_valid = 50
    train_batch_size = 50
    optimal_params_path = "opt_t" #opt_t_clever_initialisation

    # Initialize TensorBoard writer
    learning_rate = 1e-4
    run_name = f"model_lr{learning_rate}_batch{train_batch_size}"
    log_dir = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_optimal_timesteps/{run_name}"
    writer = SummaryWriter(log_dir)

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    # Initialize diffusion model components
    args = parse_arguments()
    set_seed_everything(args.seed)

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
    
    valid_dataset = LTTDataset(dir=os.path.join(data_dir, "validation"), size=num_valid, train_flag=False, use_optimal_params=True,optimal_params_path=optimal_params_path) 
    train_dataset = LTTDataset(dir=os.path.join(data_dir, "train"), size=num_train, train_flag=True, use_optimal_params=True, optimal_params_path=optimal_params_path)

    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=custom_collate_fn,
        batch_size=train_batch_size,  # Adjust batch size as needed
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=custom_collate_fn,
        batch_size=50,  # Adjust batch size as needed
        shuffle=False,
    )

    model = LTT_model(steps = steps)
    loss_fn = nn.MSELoss()#CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def calculate_lpips_loss(model, latent, img, device):
        model.eval()
        with torch.no_grad():
            outputs = model(latent)
            dis_model = DiscretizeModelWrapper(
                lambda_max=1.0,  # Adjust these values as needed
                lambda_min=0.0,
                noise_schedule=noise_schedule,
                time_mode='time'
            )
            timesteps_list = dis_model.convert(outputs)
            x_next_list = noise_schedule.prior_transformation(latent)
            x_next_computed = []
            for timestep, x_next in zip(timesteps_list, x_next_list):
                x_next = solver.sample_simple(
                    model_fn=wrapped_model,
                    x=x_next.unsqueeze(0),
                    timesteps=timestep,
                    order=order,
                    NFEs=steps,
                    condition=None,
                    unconditional_condition=None,
                    **solver_extra_params,
                )
                x_next_computed.append(x_next)
            x_next_computed = decoding_fn(torch.cat(x_next_computed, dim=0))
            lpips_loss = lpips_loss_fn(img.float(), x_next_computed.float()).mean().item()
        model.train()
        return lpips_loss

    # print(model)
    loss_list = []
    valid_loss_list = []
    valid_loss_index = []
    # Forward pass
    for i in range(1):
        print(f"\n epoch: {i}")
        for j, batch in enumerate(train_loader):
            img, latent, optimal_params = batch

            latent = latent.to(device)
            optimal_params = optimal_params.to(device)
            # optimal_params = torch.tensor([0.1140, 0.1652, 0.1298, 0.1056, 0.1084, 0.3770], device='cuda:0') 
            # optimal_params = torch.unsqueeze(optimal_params, 0).repeat(latent.size(0), 1)

            outputs = model(latent)

            # print(f"outputs: {outputs}")
            # print(f"optimal_params: {optimal_params}")
            
            loss = loss_fn(outputs, optimal_params)
            # print(f"loss: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Log training loss
            writer.add_scalar('Loss/train', loss.item(), i * len(train_loader) + j)

            if j % 50 == 0:

                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Layer: {name} | Grad Norm: {param.grad.norm().item()}")
                for batch in valid_loader:
                    with torch.no_grad():
                        img, latent, optimal_params = batch

                        latent = latent.to(device)
                        optimal_params = optimal_params.to(device)
                        #optimal_params = torch.tensor([0.1140, 0.1652, 0.1298, 0.1056, 0.1084, 0.3770], device='cuda:0') 
                        #optimal_params = torch.unsqueeze(optimal_params, 0).repeat(latent.size(0), 1)

                        outputs = model(latent)
                        loss = loss_fn(outputs, optimal_params)

                        # Log validation loss
                        writer.add_scalar('Loss/valid', loss.item(), i * len(train_loader) + j)
                        print(f"Iteration {i * len(train_loader) + j}, Validation loss: {loss.item()}")



                        #every 500 iterations, calculate lpips loss
                        if j % 500 == 0:
                            lpips_loss = calculate_lpips_loss(model, latent, img, device)
                            writer.add_scalar('Loss/LPIPS', lpips_loss, i * len(train_loader) + j)
                            print(f"Iteration {i * len(train_loader) + j}, LPIPS loss: {lpips_loss}")

            optimizer.zero_grad()

            #loss_list.append(min(loss.item(),0.1))





    # plt.plot(loss_list, label = "loss")  # Plot the loss curve
    # window_size = 100
    # rolling_window = np.convolve(loss_list, np.ones(window_size)/window_size, mode='valid')
    # plt.plot(rolling_window, label='rolling window average')
    # plt.plot(valid_loss_index, valid_loss_list, label = "valid loss")

    # plt.legend()

    # #log scale
    # # plt.yscale('log')
    # # plt.savefig(f"loss_curve_lr{learning_rate}_batch{train_batch_size}_with_dropout_0.5.png")
    # plt.savefig("PreTrained_LossCurve.png")

    # Close the TensorBoard writer
    writer.close()

    #save model 
    save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/runs/RandomModels"

    #torch.save(model.state_dict(), f"{save_path}/PreTrained.pth")
    torch.save(model.state_dict(), f"{save_path}/model_lr{learning_rate}_batch{train_batch_size}.pth")



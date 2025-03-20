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
            input_size=256,
            output_size=steps + 1,
            hidden_size=512,
            dropout=mlp_dropout
        )

        self.l1_norm = SimpleMLP.L1NormLayer()
    
    def forward(self, x):
        out = self.unet(x)
        out = self.mlp(out)
        out = F.softplus(out)
        # x = torch.sigmoid(x) 
        out = self.l1_norm(out)
        # x = torch.softmax(x, dim=1)  # Apply softmax along the class dimension

        return out

#https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
class SimpleUNet_Encoding(torch.nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.0,num_groups=32):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            # self.bn1 = nn.BatchNorm2d(out_channels)
            self.gn1 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
            self.dropout1 = nn.Dropout2d(dropout)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            # self.bn2 = nn.BatchNorm2d(out_channels)
            self.gn2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
            self.dropout2 = nn.Dropout2d(dropout)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x):
            residual = self.shortcut(x)
            # x = F.leaky_relu(self.conv1(x), 0.1) #F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
            x = F.leaky_relu(self.gn1(self.conv1(x)), 0.1)
            x = self.dropout1(x)
            # x = self.conv2(x) #self.bn2(self.conv2(x))
            x = self.gn2(self.conv2(x))
            x = self.dropout2(x)
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
        b = F.adaptive_avg_pool2d(b, (1, 1)).squeeze(-1).squeeze(-1)
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


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Add dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Add dropout
        x = self.fc3(x)

        return x


class Delta_LTT_model(nn.Module):

    def __init__(self, steps: int = 10, mlp_dropout: float = 0.0):
        super().__init__()

        self.unet = SimpleUNet_Encoding(
            in_channels=3
        )
        # Add processing for additional features
        self.mlp = SimpleMLP(
            input_size=256+1+1,
            output_size=1,
            hidden_size=512,
            dropout=mlp_dropout
        )


    def forward(self, x, current_timestep, steps_left):
        out = self.unet(x)
        out = torch.cat([out, current_timestep.unsqueeze(0).unsqueeze(0), steps_left.unsqueeze(0).unsqueeze(0)], dim=1)
        out = self.mlp(out)
        out = torch.sigmoid(out)

        return out




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


    img, latent, _ = train_dataset[0]
    img = img.to(device)
    latent = latent.to(device)
    
    print(model(latent.unsqueeze(0)))


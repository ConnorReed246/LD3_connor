import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from dataset import load_data_from_dir, LD3Dataset

# Fully connected neural network with one hidden layer
class LTT_model(nn.Module):
    def __init__(self, steps: int = 10):
        super().__init__()

        self.unet = SimpleUNet_Encoding(
            in_channels=3
        )
        self.mlp = SimpleMLP(
            input_size=256 * 8 * 8,
            output_size=steps + 1,
            hidden_size=100
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
    

    def __init__(self, input_size, output_size, hidden_size=100, dropout=0.5):
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
        x = self.l1_norm(x)
        
        # x = torch.sigmoid(x) * 2  # Scale to [0, 2]
        # x = torch.softmax(x, dim=1)  # Apply softmax along the class dimension

        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dataset
    data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'
    steps = 5
    train_size = 450
    valid_size = 50
    train_batch_size = 5

    latents, targets, conditions, unconditions, optimal_params = load_data_from_dir( # this is what we take from training, targets are original images and latents latent goal
        data_folder=data_dir, 
        limit=train_size+valid_size,
        use_optimal_params=True,
        steps=steps
    )


    def custom_collate_fn(batch):
        collated_batch = []
        for samples in zip(*batch):
            if any(item is None for item in samples):
                collated_batch.append(None)
            else:
                collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
        return collated_batch

    train_loader = DataLoader(
        LD3Dataset(
            latents[valid_size:],
            targets[valid_size:],
            conditions[valid_size:],
            unconditions[valid_size:],
            optimal_params[valid_size:],
        ),
        collate_fn=custom_collate_fn,
        batch_size=train_batch_size,  # Adjust batch size as needed
        shuffle=True,
    )

    valid_loader = DataLoader(
        LD3Dataset(
            latents[:valid_size],
            targets[:valid_size],
            conditions[:valid_size],
            unconditions[:valid_size],
            optimal_params[:valid_size],
        ),
        collate_fn=custom_collate_fn,
        batch_size=50,  # Adjust batch size as needed
        shuffle=False,
    )



    model = LTT_model(steps = steps)
    loss_fn = nn.MSELoss()#CrossEntropyLoss()
    model = model.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    # print(model)
    loss_list = []
    valid_loss_list = []
    valid_loss_index = []
    # Forward pass
    for i in range(50):
        print(f"\n epoch: {i}")
        for j, batch in enumerate(train_loader):
            img, latent, condition, uncondition, optimal_params = batch

            latent = latent.to(device)
            optimal_params = optimal_params.to(device)

            outputs = model(latent)

            # print(f"outputs: {outputs}")
            # print(f"optimal_params: {optimal_params}")
            
            loss = loss_fn(outputs, optimal_params)
            # print(f"loss: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if j % 500 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Layer: {name} | Grad Norm: {param.grad.norm().item()}")
                        
            optimizer.zero_grad()
            loss_list.append(min(loss.item(),0.1))


        for batch in valid_loader:
            with torch.no_grad():
                img, latent, condition, uncondition, optimal_params = batch

                latent = latent.to(device)
                optimal_params = optimal_params.to(device)
                outputs = model(latent)
                loss = loss_fn(outputs, optimal_params)

                print(f"outputs: {outputs[:5]}")
                print(f"optimal_params: {optimal_params[:5]}")
                print(f"loss: {loss}")
                valid_loss_list.append(min(loss.mean().item(), 0.5))
                valid_loss_index.append(i*450/train_batch_size)


    plt.plot(loss_list, label = "loss")  # Plot the loss curve
    window_size = 100
    rolling_window = np.convolve(loss_list, np.ones(window_size)/window_size, mode='valid')
    plt.plot(rolling_window, label='rolling window average')
    plt.plot(valid_loss_index, valid_loss_list, label = "valid loss")

    plt.legend()

    #log scale
    # plt.yscale('log')
    plt.savefig(f"loss_curve_lr{learning_rate}_batch{train_batch_size}_with_valid.png")


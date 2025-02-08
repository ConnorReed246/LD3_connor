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

        # self.unet = SongUNet_Encoding(
        #     img_resolution=32,                     # Image resolution at input/output.
        #     in_channels=3,                        # Number of color channels at input.
        # )
        self.unet = SimpleUNet_Encoding(
            in_channels=3
        )
        self.mlp = SimpleMLP(
            input_size=1024 * 2 * 2,
            output_size=steps + 1,
            hidden_size=100
        )
    
    def forward(self, x):
        out = self.unet(x)
        out = self.mlp(out)

        # no activation and no softmax at the end
        return out


#https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
class SimpleUNet_Encoding(torch.nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_op = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv_op(x)
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
        self.down_convolution_3 = SimpleUNet_Encoding.DownSample(128, 256)
        self.down_convolution_4 = SimpleUNet_Encoding.DownSample(256, 512)

        self.bottle_neck = SimpleUNet_Encoding.DoubleConv(512, 1024)


    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)
        return b



class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(SimpleMLP, self).__init__()
        # Define MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x, batch_size = 1):
        x = x.view(-1) #x.view(x.size(0), -1) # Flatten input from [B, C, H, W] to [B, C*H*W]
        x = F.relu(self.fc1(x))  # First layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second layer with ReLU activation
        x = self.fc3(x)  # Output layer
        x = torch.sigmoid(x) * 2  # Scale output to [0, 2] -> originally between [0, 1] #TODO STILL NEED THIS FOR NOW SINCE sometimes we output big number that will be massive when used as exponent in softmax
        x = torch.softmax(x, dim=0)  # Softmax activation since it isn't done in transformation afterwards anymore
        return x


if __name__ == "__main__":
    # Dataset
    data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'
    latents, targets, conditions, unconditions = load_data_from_dir( # this is what we take from training, targets are original images and latents latent goal
        data_folder=data_dir, 
        limit=10
    )
    ori_latents = [latent.clone() for latent in latents]

    loader = DataLoader(
        LD3Dataset(
            ori_latents,
            latents,
            targets,
            conditions,
            unconditions,
        ),
        batch_size=4,  # Adjust batch size as needed
        shuffle=True
    )

    model = LTT_model()
    print(model)

    # Forward pass
    for batch in loader:
        img, latent, ori_latent, condition, uncondition = batch
        outputs = model(latent)
        print(outputs.shape)
        print(outputs)
        break  # Remove this break to process the entire dataset

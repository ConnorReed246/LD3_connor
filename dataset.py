from typing import List, Optional, Tuple
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from gen_data import LatentGenerator


def load_data_from_dir(
    data_folder: str, limit: int = 200, use_optimal_params: bool = False, steps: int = 5
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    latents, targets, conditions, unconditions, optimal_params = [], [], [], [], []
    pt_files = [f for f in os.listdir(data_folder) if f.endswith('pt')]
    for file_name in sorted(pt_files)[:limit]: #load all training files previously created
        file_path = os.path.join(data_folder, file_name)
        data = torch.load(file_path, weights_only=True)
        latents.append(data["latent"])
        targets.append(data["img"])
        conditions.append(data.get("c", None))
        unconditions.append(data.get("uc", None))
        if use_optimal_params:
            optimal_params_path = os.path.join(data_folder, f"optimal_params_{file_name.split('_')[1].split('.')[0]}_N10_steps{steps}.pth")
            optimal_params.append(torch.load(optimal_params_path, weights_only=True)[0][0].detach()) #ignore loss and pick best params from top 10
        else:
            optimal_params.append(None)
    return latents, targets, conditions, unconditions, optimal_params


class LD3Dataset(Dataset):
    def __init__(
        self,
        latent: List[torch.Tensor],
        target: List[torch.Tensor],
        condition:  List[Optional[torch.Tensor]],
        uncondition:  List[Optional[torch.Tensor]],
        optimal_params: List[Optional[torch.Tensor]],
    ):
        self.latent = latent
        self.target = target
        self.condition = condition
        self.uncondition = uncondition
        self.optimal_params = optimal_params

    def __len__(self) -> int:
        return len(self.latent)

    def __getitem__(self, idx: int):
        img = self.target[idx]
        latent = self.latent[idx]
        condition = self.condition[idx]
        uncondition = self.uncondition[idx]
        optimal_params = self.optimal_params[idx]
        return img, latent, condition, uncondition, optimal_params
    



class LTTDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')] #check if these are sorted
        #sort these files by number
        self.image_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

        self.transform = transforms.ToTensor()
        train_flag = True if "train" in image_dir else False
        self.latent_generator = LatentGenerator(train=train_flag)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        return image, self.latent_generator.generate_latent(idx = idx)
    



 

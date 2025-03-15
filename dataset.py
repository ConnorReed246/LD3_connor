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
    def __init__(self, dir, use_optimal_params = False, size = 100, train_flag : bool = True, optimal_params_path = "opt_t"):
        self.image_dir = os.path.join(dir, "img")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.pt')]
        self.image_files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.image_files = self.image_files[:size]

        self.latent_generator = LatentGenerator(train=train_flag)

        self.use_optimal_params = use_optimal_params
        if self.use_optimal_params:
            self.opt_t_dir = os.path.join(dir, optimal_params_path)
            self.opt_t_files = [f for f in os.listdir(self.opt_t_dir) if f.endswith('.pth')]
            self.opt_t_files.sort(key=lambda x: int(x.split('_')[2]))
            self.opt_t_files = self.opt_t_files[:size]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        image = torch.load(img_path, weights_only=True)

        if self.use_optimal_params:
            opt_t_path = os.path.join(self.opt_t_dir, self.opt_t_files[idx])
            opt_t = torch.load(opt_t_path, weights_only=True)[0][0].detach()  #ignore loss and pick best params from top 10
        else:
            opt_t = None

        return image, self.latent_generator.generate_latent(idx = idx), opt_t
    



 

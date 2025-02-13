from typing import List, Optional
from dataclasses import dataclass

import torch 
from torch.utils.data import DataLoader
from torch.nn import functional as F, CrossEntropyLoss
import torchvision

import lpips 
import logging
import matplotlib.pyplot as plt
import imageio

import os 
import pickle
import numpy as np 

from dataset import LD3Dataset
from latent_to_timestep_model import LTT_model
from utils import move_tensor_to_device, compute_distance_between_two, compute_distance_between_two_L1, visual, Tensorboard_Logger,tensor_to_image

def save_gif(snapshot_path: str):
    care_files = [f for f in os.listdir(snapshot_path) if "log_best" in f]
    care_files = sorted(care_files, key=lambda f: int(f.split("_")[-1].replace(".png", "")))
    images = []
    for f in care_files:
        images.append(imageio.imread(os.path.join(snapshot_path, f)))
    imageio.mimsave(os.path.join(snapshot_path, "gif.gif"), images, duration=100.)
    print(f"Saved gif to {os.path.join(snapshot_path, 'gif.gif')}")


def custom_collate_fn(batch):
    collated_batch = []
    for samples in zip(*batch):
        if any(item is None for item in samples):
            collated_batch.append(None)
        else:
            collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    return collated_batch

@dataclass
class TrainingConfig:
    valid_data: any
    train_data: any
    train_batch_size: int
    valid_batch_size: int
    lr_time_1: float
    shift_lr: float
    shift_lr_decay: float = 0.5
    min_lr_time_1: float = 5e-5

    win_rate: float = 0.5
    patient: int = 5

    lr_time_decay: float = 0.8
    momentum_time_1: float = 0.9
    weight_decay_time_1: float = 0.0
    loss_type: str = "LPIPS"
    visualize: bool = False
    no_v1: bool = False
    prior_timesteps: Optional[List[float]] = None
    match_prior: bool = False,
    use_optimal_params: bool = False
    
@dataclass
class ModelConfig:
    net: any
    decoding_fn: any
    noise_schedule: any
    solver: any
    solver_name: str
    order: int
    steps: int
    prior_bound: float
    resolution: int
    channels: int
    time_mode: str
    solver_extra_params: Optional[dict] = None
    snapshot_path: str = "logs"
    device: Optional[str] = None

class LD3Trainer:
    def __init__(
        self, model_config: ModelConfig, training_config: TrainingConfig
    ) -> None:
        # Model parameters
        self.net = model_config.net
        self.decoding_fn = model_config.decoding_fn
        self.noise_schedule = model_config.noise_schedule
        self.solver = model_config.solver
        self.solver_name = model_config.solver_name
        self.order = model_config.order
        self.steps = model_config.steps
        self.prior_bound = model_config.prior_bound
        self.resolution = model_config.resolution
        self.channels = model_config.channels
        self.time_mode = model_config.time_mode

        # Learning rate parameters
        self.lr_time_1 = training_config.lr_time_1
        self.shift_lr = training_config.shift_lr
        self.shift_lr_decay = training_config.shift_lr_decay
        self.min_lr = training_config.min_lr_time_1
        self.lr_time_decay = training_config.lr_time_decay
        self.momentum_time_1 = training_config.momentum_time_1
        self.weight_decay_time_1 = training_config.weight_decay_time_1

        # Training data and batch sizes
        self.train_data = training_config.train_data
        self.valid_data = training_config.valid_data
        self.train_batch_size = training_config.train_batch_size
        self.valid_batch_size = training_config.valid_batch_size
        self._create_valid_loaders()
        self._create_train_loader()
        self.eval_on_one = False #Maybe change back, but probably better to leave it to keep loss consistent
        self.use_optimal_params = training_config.use_optimal_params

        # Training state
        self.cur_iter = 0
        self.cur_round = 0
        self.count_worse = 0
        self.count_min_lr_hit = 0
        self.best_loss = float("inf")

        # Other parameters
        self.patient = training_config.patient
        self.no_v1 = training_config.no_v1
        self.win_rate = training_config.win_rate
        self.snapshot_path = model_config.snapshot_path
        os.makedirs(self.snapshot_path, exist_ok=True)
        self.visualize = training_config.visualize
        if Tensorboard_Logger.writer_exists(): 
            self.writer = Tensorboard_Logger.get_writer()

        # Device and optimizer setup
        self._set_device(model_config.device)
        self.params = self._initialize_params()

        self.ltt_model = LTT_model(steps=self.steps)
        self.ltt_model = self.ltt_model.to(self.device)

        self.optimizer = torch.optim.Adam(self.ltt_model.parameters(), lr=training_config.lr_time_1) #TODO maybe add momentum an weight decay?    momentum=training_config.momentum_time_1,  weight_decay=training_config.weight_decay_time_1,

        self.prior_timesteps = training_config.prior_timesteps
        self.match_prior = training_config.match_prior

        # Additional attributes
        self.solver_extra_params = model_config.solver_extra_params or {}
        self.lambda_min = self.noise_schedule.lambda_min
        self.lambda_max = self.noise_schedule.lambda_max
        self.time_max = self.noise_schedule.inverse_lambda(self.lambda_min)
        self.time_min = self.noise_schedule.inverse_lambda(self.lambda_max)

        # Initialize baseline, what does this do?
        # self._compute_baseline()

        # Initialize loss function
        self.loss_type = training_config.loss_type #LPIPS -> differnece in two images 
        self.loss_fn = self._initialize_loss_fn()
        self.loss_fn_optimal_params = CrossEntropyLoss()
        self.loss_vector = None




    #Nothing here is updated to LTT
    # def _train_to_match_prior(self, prior_timesteps=None):
    #     if prior_timesteps is None:
    #         prior_timesteps = self.prior_timesteps
            
    #     if prior_timesteps is None:
    #         return 
    #     logging.info(f"Matching prior timesteps")
    #     prior_timesteps = self.noise_schedule.inverse_lambda(-np.log(prior_timesteps)).to(self.device).float()
        
    #     dis_model = discretize_model_wrapper(
    #         self.params,
    #         self.params2,
    #         self.lambda_max,
    #         self.lambda_min,
    #         self.noise_schedule,
    #         self.time_mode,
    #         self.win_rate,
    #     )
        
    #     self.params.requires_grad = True
    #     self.params2.requires_grad = False
        
    #     loss_time = float("inf")
    #     while loss_time > 1e-3:
    #         self.optimizer_lamb1.zero_grad()
    #         self.optimizer_lamb2.zero_grad()
    #         times1, times2 = dis_model()
    #         loss_time = (times1 - prior_timesteps).pow(2).mean()
    #         logging.info(f"Loss time: {loss_time}")
    #         loss_time.backward()
    #         self.optimizer_lamb1.step()
        
    def _initialize_loss_fn(self):
        if self.loss_type == 'LPIPS':
            return lpips.LPIPS(net='vgg').to(self.device)
        elif self.loss_type == 'L2':
            return lambda x, y : compute_distance_between_two(x, y, self.channels, self.resolution)
        elif self.loss_type == 'L1':
            return lambda x, y: compute_distance_between_two_L1(x, y, self.channels, self.resolution)
        else:
            raise NotImplementedError
    
    def _initialize_params(self):
        params = torch.nn.Parameter(torch.ones(self.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)
        return params

    def _set_device(self, device):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_valid_loaders(self):
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.train_batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.valid_only_loader = DataLoader(self.valid_data, batch_size=self.valid_batch_size, shuffle=False, collate_fn=custom_collate_fn)

    def _create_train_loader(self):
        self.train_loader = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    def _solve_ode(self, img=None, latent=None, optimal_param = None, condition=None, uncondition=None, valid=False, use_optimal_params: bool = False): #TODO remove timestep
        batch_size = latent.shape[0]
        latent = latent.reshape(batch_size, self.channels, self.resolution, self.resolution) 

        with torch.set_grad_enabled(not valid):
            params_list = self.ltt_model(latent)

            if use_optimal_params and not valid: #this is only run on training data when we want to match optimal params
                self.loss_vector = self.loss_fn_optimal_params(params_list, optimal_param)
                loss = self.loss_vector.mean()
                return loss, None, None

            else: #this is the normal case
                dis_model = DiscretizeModelWrapper( #Changed through LTT
                lambda_max=self.lambda_max,
                lambda_min=self.lambda_min,
                noise_schedule=self.noise_schedule,
                time_mode = self.time_mode,
                )

                timesteps_list = dis_model.convert(params_list)
                self.timesteps_list = timesteps_list

                x_next_list = self.noise_schedule.prior_transformation(latent) #Multiply with timestep in edm case (x80 in beginning)

                x_next_computed = []
                for timestep, x_next in zip(timesteps_list, x_next_list):
                    x_next = self.solver.sample_simple(
                        model_fn=self.net,
                        x=x_next.unsqueeze(0),
                        timesteps=timestep,
                        order=self.order,
                        NFEs=self.steps,
                        condition=condition,
                        unconditional_condition=uncondition,
                        **self.solver_extra_params,
                    )
                    x_next_computed.append(x_next)#This was wrong the whole time?

                x_next_computed = self.decoding_fn(torch.cat(x_next_computed, dim=0))                
                self.loss_vector = self.loss_fn(img.float(), x_next_computed.float()).squeeze()
                loss = self.loss_vector.mean()
                logging.info(f"{self._current_version} Loss: {loss.item()}")

                return loss, x_next_computed.float(), img.float()
        

    @property
    def _current_version(self):
        return 'Ver1' if self._is_in_version_1() else 'Ver2'

    def _is_in_version_1(self):
        return self.cur_round < self.training_rounds_v1

    def _compute_baseline(self): #just for visualisation
        self.straight_line = torch.linspace(self.lambda_min, self.lambda_max, self.steps + 1)
        self.time_logSNR = self.noise_schedule.inverse_lambda(self.straight_line).to(self.device)        
        time_max = self.noise_schedule.inverse_lambda(self.lambda_min)
        time_min = self.noise_schedule.inverse_lambda(self.lambda_max)
        self.time_s = torch.linspace(time_max.item(), time_min.item(), 1000)
        self.time_straight = torch.linspace(time_max.item(), time_min.item(), self.steps + 1)
        self.time_straight = self.time_straight.to(self.device)
        self.straight_time = self.noise_schedule.marginal_lambda(self.time_s)
        t_order = 2
        self.time_q = torch.linspace((time_max**(1/t_order)).item(), (time_min**(1/t_order)).item(), 1000)**t_order
        self.quadratic_time = torch.linspace((time_max**(1/t_order)).item(), (time_min**(1/t_order)).item(), self.steps + 1)**t_order

        self.quadratic_time = self.quadratic_time.to(self.device)
        self.time_quadratic = self.noise_schedule.marginal_lambda(self.time_q)
        # time_edm 
        self.time_edm = self.solver.get_time_steps('edm', time_max.item(), time_min.item(), 999, self.device)
        self.lambda_edm = self.noise_schedule.marginal_lambda(self.time_edm)
        
    def _run_validation(self):
        total_loss = 0.
        count = 0
        outputs = list()
        targets = list()
        with torch.no_grad():
            for img, latent, condition, uncondition, optimal_param in self.valid_only_loader:
                # condition = condition.squeeze()
                # uncondition = uncondition.squeeze()
                img = img.to(self.device)
                latent = latent.to(self.device).reshape(latent.shape[0], -1)
                # if condition is not None:
                #     condition = condition.to(self.device)
                # if uncondition is not None:
                #     uncondition = uncondition.to(self.device)
                loss, output, target = self._solve_ode(img=img, latent=latent, optimal_param=optimal_param, condition=condition, uncondition=uncondition, valid=True, use_optimal_params=self.use_optimal_params) #TODO here we outpus output and target and anaylse the differnce?

                total_loss += loss.item()
                count += 1
                outputs.append(output)
                targets.append(target)

                if self.eval_on_one: #we only evaluate on 10 validation steps like this TODO: maybe change?
                    break 
                    
        output = torch.cat(outputs, dim=0)
        target = torch.cat(targets, dim=0)
        return total_loss / count, output, target
    
    def _visual_times(self) -> None:
        """

            Visualize time discretization of baselines and ours
        """

        log_path = os.path.join(self.snapshot_path, f"log_best_{self.cur_iter}.png")

        plt.plot(self.logSNR1.cpu().numpy(), 'o', label="Our discretization1")
        plt.plot(self.logSNR2.cpu().numpy(), 'x', label="Our discretization2")
        x_axis = np.linspace(0, self.steps, self.steps + 1)
        plt.plot(x_axis, self.straight_line.cpu().numpy(), label="Baseline logSNR")
        x_axis = np.linspace(0, self.steps, 1000)            
        plt.plot(x_axis, self.straight_time.cpu().numpy(), label="Baseline time uniform")
        plt.plot(x_axis, self.time_quadratic.cpu().numpy(), label="Baseline time quadratic")
        plt.plot(x_axis, self.lambda_edm.cpu().numpy(), label="Baseline time edm")

        # draw a horizontal line at low_t_lambda
        plt.xlabel("Reverse step i")
        plt.ylabel("LogSNR(t_i)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path)
        plt.close()

    def _save_checkpoint(self):

        torch.save(self.ltt_model.state_dict(), self.snapshot_path + "/ltt_model.pt")
        # save dataloader, valid_loader, valid_only_loader
        pickle.dump(self.train_data, open(os.path.join(self.snapshot_path, "train_data.pkl"), "wb"))
        pickle.dump(self.valid_data, open(os.path.join(self.snapshot_path, "valid_data.pkl"), "wb"))

        # model must be created again with parameters
    

    
    def _load_checkpoint(self, reload_data:bool):
        state_dict = torch.load(self.snapshot_path + "/ltt_model.pt", weights_only=True)
        self.ltt_model.load_state_dict(state_dict)  # Load the model state
        if reload_data:
            self.train_data = pickle.load(open(os.path.join(self.snapshot_path, "train_data.pkl"), "rb"))
            self.valid_data = pickle.load(open(os.path.join(self.snapshot_path, "valid_data.pkl"), "rb"))
            self._create_train_loader()
            self._create_valid_loaders()

    def _examine_checkpoint(self, iter: int) -> None:
        logging.info(f"{self._current_version} Saving snapshot at iter {iter}")
        total_loss, output, target = self._run_validation() # get loss, output and target of validation set
        
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.count_worse = 0
            self._save_checkpoint() #saves self.params.data and best_t_steps
            #self._visual_times() TODO: maybe revisualize with average timsteps or distributions
            #save_gif(self.snapshot_path)
        else:
            self.count_worse += 1
            logging.info(f"{self._current_version} Count worse: {self.count_worse}")
        
        logging.info(f"{self._current_version} Validation loss: {total_loss}, best loss: {self.best_loss}")
        logging.info(f"{self._current_version} Iter {iter} snapshot saved!")
        
        ##################### Tensorboard logging #####################

        # layout = {
        #     "Timesteps": {
        #         "timestep": ["Multiline", ["timestep/" + str(i) for i in range(len(self.t_steps1))]],
        #     },
        # }
        # self.writer.add_custom_scalars(layout) #TODO add visualisation for new timesteps

        if self.writer:
            self.writer.add_scalar(f"Validation/Loss", total_loss, iter)
            self.writer.add_scalar(f"Validation/Best_Loss", self.best_loss, iter)

            if iter == 0:
                grid = torchvision.utils.make_grid(tensor_to_image(target[:8]))
                self.writer.add_image('Validation/Target', grid, iter)
            # if (iter % 5 == 0):
            #     # visual(torch.cat([output[:8], target[:8]], dim=0), os.path.join(self.snapshot_path, f"learned_newnoise_ep{iter}.png"), img_resolution=self.resolution)
            #     grid = torchvision.utils.make_grid(tensor_to_image(output[:8]))
            #     self.writer.add_image('Validation/Output', grid, iter)
            if self.count_worse == 0:
                grid = torchvision.utils.make_grid(tensor_to_image(output[:8]))
                self.writer.add_image('Validation/Output', grid, iter)

                # for i in range(len(self.t_steps1)):
                #     self.writer.add_scalar(f"timestep/{i}", self.t_steps1[i], iter)
                self.writer.add_histogram("Timesteps/Distribution", self.timesteps_list, iter)


        ###############################################################
            
        
        if self.count_worse >= self.patient:
            logging.info(f"{self._current_version} Loading best model")
            self._load_checkpoint(reload_data=True)
            self.count_worse = 0

            if self.eval_on_one: 
                self.eval_on_one = False
                self.best_loss, _, _ = self._run_validation()
                logging.info("Start evaluation on all valid set from now. Not decay learning rate.")
                return 

            self.optimizer.param_groups[0]['lr'] = max(self.lr_time_decay * self.optimizer.param_groups[0]['lr'], self.min_lr) #IF patience reached we also decay learning rate?
            logging.info(f"{self._current_version} Decay time1 lr to {self.optimizer.param_groups[0]['lr']}")

        
            if self.optimizer.param_groups[0]['lr'] <= self.min_lr: #TODO this might break
                self.count_min_lr_hit += 1

    def _set_trainable_params(self, is_train:bool, is_no_v1:bool)->None:
        if is_train:
            self.params.requires_grad = True
                          
        else:
            self.params.requires_grad = False

    def _log_valid_distance(self, ori_latent: torch.tensor, latent: torch.tensor):
        assert ori_latent.shape == latent.shape, "Shape of ori_latent and latent mismatched"
        sq = (latent.reshape(latent.shape[0], -1) - ori_latent.reshape(latent.shape[0], -1)).pow(2)
        distances = sq.sum(dim=1).sqrt().detach().cpu().numpy()
        logging.info(f"{self._current_version} Distance: {distances}")

    def _update_dataloader(self, ori_latents:List[torch.tensor], 
                           latents:List[torch.tensor], 
                           targets:List[torch.tensor], 
                           conditions: List[Optional[torch.tensor]],
                           unconditions: List[Optional[torch.tensor]],
                           is_train:bool):
        custom_train_dataset = LD3Dataset(ori_latents, latents, targets, conditions, unconditions)
        if is_train:
            self.train_data = custom_train_dataset
            self._create_train_loader()
        else:
            self.valid_data = custom_train_dataset
            self._create_valid_loaders()
        exit() #TODO this doesn't work and should never be called

    def _update_latents(self, latent, condition, uncondition, ori_latent, img, latent_params, loss_vector_ref, prior_bound):
        parameter_data_detached = latent_params.detach()
        cloned_ori_latent = ori_latent.clone()
        diff = parameter_data_detached.data - cloned_ori_latent
        diff_norm = diff.norm(dim=1, keepdim=True)
        pass_bound = diff_norm > prior_bound
        pass_bound = pass_bound.flatten()
        parameter_data_detached.data[pass_bound] = cloned_ori_latent[pass_bound] + prior_bound * diff[pass_bound] / diff_norm[pass_bound]
        
        _, _, _ = self._solve_ode(img=img, latent=parameter_data_detached.data, condition=condition, uncondition=uncondition, valid=False)
        
        to_update_mask =  self.loss_vector < loss_vector_ref
        parameter_data_detached.data = parameter_data_detached.data.reshape(-1, self.channels, self.resolution, self.resolution)
        latent[to_update_mask] = parameter_data_detached.data[to_update_mask]
        return latent, to_update_mask

    def _train_one_round(self):
        no_change = True
        logging.info(f"{self._current_version} Round {self.cur_round}")

        if self.cur_round > 0:
            self._load_checkpoint(reload_data=False)
            self.count_worse = 0
        
        self._examine_checkpoint(self.cur_iter) # run validation on current latent and time steps

        for loader_idx, loader in enumerate([self.train_loader]): #TODO validation at end of round for now since it happens during anyway , self.valid_loader
            
            if loader_idx == 1: 
                if self.prior_bound == 0.0:
                    print("skipped validation?")
                    continue
                valid = True

            else:
                valid = False
            
            # latents, targets, conditions, unconditions = [], [], [], []
            for img, latent, condition, uncondition, optimal_param in loader: #1 at a time in validation

                img, latent, condition, uncondition, optimal_param = move_tensor_to_device(img, latent, condition, uncondition, optimal_param, device=self.device)
                
                # Flattent latents
                batch_size = latent.shape[0]
                
                ############################## TENSORBOARD ##############################
                if loader_idx == 0 and self.cur_iter == 0:
                    self.writer.add_graph(self.ltt_model, latent[:1])
                ########################################################################

                latent = latent.reshape(batch_size, -1) # torch.Size([1, 3072])
                loss, _, _ = self._solve_ode(img=img, latent=latent, optimal_param=optimal_param, condition=condition, uncondition=uncondition, valid=valid, use_optimal_params = self.use_optimal_params)

                if loader_idx == 0:

                    loss.backward()
                    logging.info(f"{self._current_version} Iter {self.cur_iter} {'Train' if loader_idx == 0 else 'Val'} Loss: {loss.item()}")
                    self.writer.add_scalar(f"Train/Loss", loss.item(), self.cur_iter) #TENSORBOARD
                    # torch.nn.utils.clip_grad_norm_(self.params, 1.0) this does nothing since we aren't upd

                    #TODO we have to rewrite this so that self.ltt model is backpropagated
                    self.optimizer.step() #does this ever change?
                    ##################### TENSORBOARD ##################### #TODO update to new gradients?
                    # for i, value in enumerate(self.optimizer_lamb1.param_groups[0]["params"][0].grad):
                    #     self.writer.add_scalar(f"Gradients/{i}", value, self.cur_iter)
                    #######################################################
                    self.optimizer.zero_grad()


                    self.cur_iter += 1
                    self._examine_checkpoint(self.cur_iter) # evaluate
        


                # latent = latent.reshape(-1, self.channels, self.resolution, self.resolution).detach().cpu()
                # img = img.detach().cpu()
                # condition = condition.detach().cpu() if condition is not None else None
                # uncondition = uncondition.detach().cpu() if uncondition is not None else None
                
                # for j in range(latent.shape[0]):
                #     targets.append(img[j])
                #     latents.append(latent[j])
                #     conditions.append(condition[j] if condition is not None else None)
                #     unconditions.append(uncondition[j] if uncondition is not None else None)
            
        return no_change, False
        
    def train(self, training_rounds_v1) -> None:
        
        total_round = training_rounds_v1 
        self.training_rounds_v1 = training_rounds_v1

        # if self.match_prior:
        #     self._train_to_match_prior()

        while self.cur_round < total_round:
            no_latent_change, should_stop = self._train_one_round()
            if should_stop:
                return
            self.cur_round += 1
            
            #learning rate decreases with time TODO maybe remvove this?
            if no_latent_change and self.prior_bound > 0:
                self.shift_lr *= self.shift_lr_decay
        
        logging.info(f"{self._current_version} Max round reached, stopping")
        torch.save(self.ltt_model.state_dict(), self.snapshot_path + "/final_ltt_model.pt")



class DiscretizeModelWrapper:
    '''
    Class added through LTT that allows dymanic adaption of params
    '''

    def __init__(self, lambda_max, lambda_min, noise_schedule, time_mode):
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.noise_schedule = noise_schedule
        self.time_mode = time_mode

    
    def model_time_fn(self, input1):
        time1 = input1
        t_max, t_min = self.noise_schedule.inverse_lambda(self.lambda_min).to(time1.device), self.noise_schedule.inverse_lambda(self.lambda_max).to(time1.device)
        time_plus = torch.nn.functional.softmax(time1, dim=0)
        exit() #TODO WE NEED TO ADAPT THIS BEFORE USING IT
        time_md = torch.cumsum(time1, dim=0).flip(0)
        normed = (time_md - time_md[-1]) / (time_md[0] - time_md[-1])
        time_steps = normed * (t_max - t_min) + t_min
        mask = torch.ones_like(normed)
        mask[0] = 0.
        mask[-1] = 0.
        return time_steps

    def model_lambda_fn(self, input1):
        lambda1 = input1  # Shape: [batch_size, num_steps+1]
        # Cumulative sum along the time dimension (dim=1)
        lamb_md = torch.cumsum(lambda1, dim=1)  # Now keeps batch dimension
        
        # Normalize per sample in the batch
        min_vals = lamb_md.min(dim=1, keepdim=True).values
        max_vals = lamb_md.max(dim=1, keepdim=True).values
        normed = (lamb_md - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid division by zero
        
        # Scale to lambda range
        lamb_steps1 = normed * (self.lambda_max - self.lambda_min) + self.lambda_min
        time1 = self.noise_schedule.inverse_lambda(lamb_steps1)
        return time1


    def convert(self, input1):
        return self.model_time_fn(input1) if self.time_mode == 'time' else self.model_lambda_fn(input1)




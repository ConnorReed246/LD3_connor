import torch 
import pickle
from noise_schedulers import NoiseScheduleVE
from torch.utils.checkpoint import checkpoint 

def model_wrapper(model, noise_schedule, return_bottleneck, class_labels=None, use_checkpoint=False):
    '''
    always return a model that predicting noise!
    '''
    if return_bottleneck:
        def hook_fn(module, input, output):
            global bottleneck_output
            bottleneck_output = output
    
        model.model.enc["8x8_block3"].affine.register_forward_hook(hook_fn)

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = t_continuous
        if use_checkpoint:
            output = checkpoint(model, x, t_input, cond)
        else:
            output = model(x, t_input, cond)
        alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
        
        if return_bottleneck:
            return ((x - alpha_t[:, None, None, None] * output) / sigma_t[:, None, None, None]).to(torch.float32), bottleneck_output.to(torch.float32) #TODO this might break
        else:
            return ((x - alpha_t[:, None, None, None] * output) / sigma_t[:, None, None, None]).to(torch.float32)

    def model_fn(x, t_continuous, *args, **kwargs):
        return noise_pred_fn(x, t_continuous, class_labels)

    return model_fn

def get_pretrained_sde_model(args, return_bottleneck, requires_grad=False):
    '''
    checked!
    '''
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(args.ckp_path, "rb") as f:
        #load the pytorch model
        net = pickle.load(f)["ema"].to(device)
    
    #set gradient tracking to false to avoid unnecessary computation
    if not requires_grad:
        for param in net.parameters():
            param.requires_grad = False
    noise_schedule = NoiseScheduleVE(schedule='edm')
    return model_wrapper(net, noise_schedule, return_bottleneck), net, lambda x: x, noise_schedule, net.img_resolution, net.img_channels, net.img_resolution, net.img_channels
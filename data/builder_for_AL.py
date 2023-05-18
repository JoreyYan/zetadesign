import os, sys
import json, time
# import yaml
from easydict import EasyDict
# online package
import torch
import numpy as np
# optimizer
import torch.optim as optim






#lr
from timm.scheduler import CosineLRScheduler

#dl


# def cfg_from_yaml_file(cfg_file):
#     config = EasyDict()
#     with open(cfg_file, 'r') as f:
#
#         try:
#             new_config = yaml.load(f, Loader=yaml.FullLoader)
#         except:
#             new_config = yaml.safe_load(f)
#     merge_new_config(config=config, new_config=new_config)
#     return config
def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config
def save_checkpoint(base_model, optimizer_CLIP,epoch,  prefix, args, logger = None):

    torch.save({
                'base_model' : base_model.state_dict() ,
                'optimizer' : optimizer_CLIP.state_dict(),

                'epoch' : epoch,
                # 'metrics' : val_metrics.state_dict() if val_metrics is not None else dict(),

                }, os.path.join(args.experiment_path, prefix + '.pth'))

def build_opti_sche(base_model,
                    optimizer,
                    scheduler,**kwargs
                    ):
    opti_config = optimizer
    if opti_config['type'] == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config['type']  == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config['kwargs'])
    elif opti_config['type']  == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config['kwargs'])
    else:
        raise NotImplementedError()

    sche_config = scheduler

    if sche_config['type'] == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config['kwargs']['epochs'],
                                      #t_mul=1.0,
                                      lr_min=1e-5,
                                      #decay_rate=0.1,
                                      warmup_lr_init=1e-4,
                                      warmup_t=sche_config['kwargs']['initial_epochs'],
                                      cycle_limit=1,
                                      t_in_epochs=True)
    elif sche_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config['kwargs'])
    elif sche_config['type'] == 'function':
        scheduler = None
    else:
        raise NotImplementedError()



    return optimizer, scheduler

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')

    print(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])






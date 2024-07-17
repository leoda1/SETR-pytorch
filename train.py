# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/07/17 11:28:43
@Project -> : SETR-pytorch$
==================================================================================
"""

import os
import numpy as np
import torch
import torch.distributed as dist

from nets.SETR import SETR
from utils.utils import seed_everything, download_weights

if __name__ == "__main__":
    #set
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    num_classes = 12

    #model
    backbone = 'setr'
    pretrained = False
    model_path = ""
    input_shape = [512, 512]

    #train
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 6
    UnFreeze_Epoch = 100
    UnFreeze_batch_size = 3
    VOCdevkit_path = 'VOCdevkit'

    #parameters
    Init_lr = 7e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = ""
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_step = 'cos'

    #save
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10

    #loss functions
    dice_loss = False
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4
    seed_everything(seed)
    ngpus_per_node  = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, "
                  f"local_rank = "f"{local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)
#================================================================================
#   这里model可选择模型又SETR和VIT
#   model = SETR(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
#   model = VIT(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
#================================================================================
    model = SETR(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)

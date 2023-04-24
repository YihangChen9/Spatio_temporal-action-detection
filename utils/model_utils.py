import os
import hashlib
import requests
from tqdm import tqdm
import torch.nn as nn

import torch


def load_detr_weights(model, pretrain_dir, cfg):
    '''
        load the weight from the pre-train Detr model to accelerate the training process
        only used for AVA dataset
    '''

    # load parameter from the pre-train model
    checkpoint = torch.load(pretrain_dir, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {}

    # save the parameter into a dict
    for k, v in checkpoint['model'].items():
        if k.split('.')[1] == 'transformer':
            pretrained_dict.update({k: v})
        elif k.split('.')[1] == 'bbox_embed':
            pretrained_dict.update({k: v})
        elif k.split('.')[1] == 'query_embed':
            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM * (cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE)
            else:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM 
            pretrained_dict.update({k: v[:query_size]})

    # fit the parameter of detr into TubeR
    pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    unused_dict = {k: v for k, v in pretrained_dict.items() if not k in model_dict}
    print("detr unused model layers:", unused_dict.keys())
    model_dict.update(pretrained_dict_)

    # pass the parameter into model
    model.load_state_dict(model_dict)
    print("load pretrain success")


def deploy_model_ava(model, cfg, is_tuber=True):
    """
    Deploy model to multiple GPUs for DDP training.
    """
    # deploy the model on all available GPU
    if cfg.DDP_CONFIG.DISTRIBUTED:
        if cfg.DDP_CONFIG.GPU is not None:
            torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
            model.cuda(cfg.DDP_CONFIG.GPU)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[cfg.DDP_CONFIG.GPU],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.DDP_CONFIG.GPU is not None:
        torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
        model = model.cuda(cfg.DDP_CONFIG.GPU)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # load weight of Detr for training on AVA
    print("loading detr")
    load_detr_weights(model, cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR, cfg)

    return model


def deploy_model_jhmdb(model, cfg, is_tuber=True):
    """
    Deploy model to multiple GPUs for DDP training.
    """
    # deploy the model on all available GPU
    if cfg.DDP_CONFIG.DISTRIBUTED:
        if cfg.DDP_CONFIG.GPU is not None:
            torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
            model.cuda(cfg.DDP_CONFIG.GPU)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[cfg.DDP_CONFIG.GPU],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif cfg.DDP_CONFIG.GPU is not None:
        torch.cuda.set_device(cfg.DDP_CONFIG.GPU)
        model = model.cuda(cfg.DDP_CONFIG.GPU)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    return model



def load_model(model, cfg, load_fc=True):
    """
    Load pretrained model weights.
    """
    # get weight of trained model
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        print("=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
        if cfg.DDP_CONFIG.GPU is None:
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.DDP_CONFIG.GPU)
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=loc)
        model_dict = model.state_dict()
        if not load_fc:
            del model_dict['module.fc.weight']
            del model_dict['module.fc.bias']

        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        unused_dict = {k: v for k, v in checkpoint['model'].items() if not k in model_dict}
        not_found_dict = {k: v for k, v in model_dict.items() if not k in checkpoint['model']}

        print("unused model layers:", unused_dict.keys())
        print("not found layers:", not_found_dict.keys())
        print(checkpoint['model']['module.query_embed.weight'].size())

        # load the weight into model 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cfg.CONFIG.MODEL.PRETRAINED_PATH, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))

    return model, None



def save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler):
    '''
        save the checkpoint of model parameter during the traing process
    '''
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': cfg}

    model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('Saving model at epoch %d to %s' % (epoch, model_save_dir))
    save_path = os.path.join(model_save_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)

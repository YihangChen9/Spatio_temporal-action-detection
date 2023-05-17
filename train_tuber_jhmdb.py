import argparse
import datetime
import time

import torch
import torch.optim
from tensorboardX import SummaryWriter

from models.tuber import build_model
from utils.model_utils import deploy_model_jhmdb, load_model, save_checkpoint
from utils.video_action_recognition import validate_tuber_jhmdb_detection, train_tuber_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir
from datasets.jhmdb_frame import build_dataloader
from utils.lr_scheduler import build_scheduler
import os


def main_worker(cfg):
    '''
        This part is the main pipeline to train the TubeR model
        Including load the dataset, initial the model, criterion, post processor, 
        optimizer and learning rate scheduler and run the train method
    '''
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None

    # create model and deploy the model on GPU
    print('Creating TubeR model: %s' % cfg.CONFIG.MODEL.NAME)
    model, criterion, postprocessors = build_model(cfg)
    model = deploy_model_jhmdb(model, cfg, is_tuber=True)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)

    print("test sampler", len(train_loader))
    # create criterion
    criterion = criterion.cuda()

    # parameter dictionnary, set the learning rate for different parameters
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and "class_embed" not in n and "query_embed" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model.named_parameters() if "class_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR,
        },
        {
            "params": [p for n, p in model.named_parameters() if "query_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR,
        },
    ]

    # create optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.CONFIG.TRAIN.LR, weight_decay=cfg.CONFIG.TRAIN.W_DECAY)

    # create lr scheduler
    if cfg.CONFIG.TRAIN.LR_POLICY == "step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
    else:
        lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    # option to load the parameter in checkpoint to continue training
    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)
    
    print('Start training...')
    start_time = time.time()
    max_accuracy = 0.0

    # run the train program and save the checkpoint of parameter each epoch
    for epoch in range(0 , cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)
        train_tuber_detection(cfg, model, criterion, train_loader, optimizer, epoch, cfg.CONFIG.LOSS_COFS.CLIPS_MAX_NORM, lr_scheduler, writer)
        save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler)

    # validate the performance of the trained model
    validate_tuber_jhmdb_detection(cfg, model, criterion, postprocessors, val_loader, 0, writer)

    # close the tensorboard writer
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        writer.close()

    # calculate the training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # get configuration file
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='configuration/Tuber_CSN152_JHMDB.yaml',
                        help='path to config file.')
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    # using multi process to run the main worker
    spawn_workers(main_worker, cfg)

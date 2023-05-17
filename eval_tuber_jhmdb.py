import argparse
import datetime
import time

import torch
import torch.optim
from tensorboardX import SummaryWriter

from models.tuber import build_model
from utils.model_utils import deploy_model_jhmdb, load_model
from utils.video_action_recognition import validate_tuber_jhmdb_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir
from datasets.jhmdb_frame import build_dataloader
from utils.lr_scheduler import build_scheduler
import os


def main_worker(cfg):
    '''
        This part is the main pipeline to train the TubeR model
        Including load the dataset and model then run the evaluate method
    '''
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None

    # create model and deploy on GPUs
    print('Creating TubeR model: %s' % cfg.CONFIG.MODEL.NAME)
    model, criterion, postprocessors = build_model(cfg)
    model = deploy_model_jhmdb(model, cfg, is_tuber=True)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create dataset and dataloader
    _, val_loader, _, val_sampler, _ = build_dataloader(cfg)
    print("test sampler", len(val_loader))

    # create criterion
    criterion = criterion.cuda()

    # load the trained parameter from the checkpoint file
    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)

    # run the evaluate method
    print('Start evaluating...')
    start_time = time.time()
    validate_tuber_jhmdb_detection(cfg, model, criterion, postprocessors, val_loader, 0, writer)
    print("eval finished")

    # calculate the time for evaluation
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('testing time {}'.format(total_time_str))

    # close the tensorboard writer
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        writer.close()


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

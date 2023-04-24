# pylint: disable=line-too-long
"""
Utility functions for task
"""
import glob
import json
import os
import time
import numpy as np

import torch
import math

from .utils import AverageMeter
from evaluates.evaluate_ava import STDetectionEvaluater, STDetectionEvaluaterSinglePerson
from evaluates.evaluate_jhmdb import STDetectionEvaluaterJHMDB


def train_tuber_detection(cfg, model, criterion, data_loader, optimizer, epoch, max_norm, lr_scheduler, writer=None):
    '''
        this is the traning method of TubeR
    '''

    # initialize the scalar used in train process, 
    # AverageMeter is a class for keeping track of averages and value counts.
    batch_time = AverageMeter() # time to process each batch
    data_time = AverageMeter() # time to load data for each batch
    class_err = AverageMeter() 
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    end = time.time()

    # ser model and criterion to train mode
    model.train()
    criterion.train()

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    #  iterate the train set to train the model
    for idx, data in enumerate(data_loader):
        
        # calcalate the time of train process
        data_time.update(time.time() - end)

        # set device to GPU
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)

        # get image information from dataloader
        samples = data[0]

        # get target from the dataloader
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)
         
        for t in targets: del t["image_id"]
        
        # move sample and target to GPU
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # get the output value from model
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                outputs = model(samples)

        # chenck whether the output is invalid
        if not math.isfinite(outputs["pred_logits"][0].data.cpu().numpy()[0,0]):
            print(outputs["pred_logits"][0].data.cpu().numpy())
        
        # calculate the each part loss function 
        loss_dict = criterion(outputs, targets)

        # update the weight of each part in loss function
        weight_dict = criterion.weight_dict
        if epoch > cfg.CONFIG.LOSS_COFS.WEIGHT_CHANGE:
            weight_dict['loss_ce'] = cfg.CONFIG.LOSS_COFS.LOSS_CHANGE_COF

        # sum each part of loss function and get the loss of the model
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # set the gradient to zero
        optimizer.zero_grad()

        # back propagation the losses
        losses.backward()

        # use gradient clipping to avoid gradient exploding
        if max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # update the gradient
        optimizer.step()

        # update the learning rate
        if cfg.CONFIG.TRAIN.LR_POLICY == 'cosine':
            lr_scheduler.step_update(epoch * len(data_loader) + idx)

        # update batch time
        batch_time.update(time.time() - end)
        end = time.time()

        # print the training detail of this epoch
        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
            print(print_string)
            for param in optimizer.param_groups:
                lr = param['lr']
            print('lr: ', lr)

            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)

            # reduce the loss on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            # update the loss
            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))

            # if loss is infinite, stop training
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                exit(1)

            # print the loss at this epoch
            print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}, loss_ce_b: {loss_ce_b:.3f}'.format(
                class_error=class_err.avg,
                loss=losses_avg.avg,
                loss_bbox=losses_box.avg,
                loss_giou=losses_giou.avg,
                loss_ce=losses_ce.avg,
                loss_ce_b=losses_ce_b.avg,
                # cardinality_error=loss_dict_reduced['cardinality_error']
            )
            print(print_string)

            # write loss information into tensorboard to visualization the trainning process
            writer.add_scalar('train/class_error', class_err.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/totall_loss', losses_avg.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_bbox', losses_box.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_giou', losses_giou.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_ce', losses_ce.avg, idx + epoch * len(data_loader))
            writer.add_scalar('train/loss_ce_b', losses_ce_b.avg, idx + epoch * len(data_loader))



@torch.no_grad()
def validate_tuber_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer):
    '''
        this function is used to evaluate the AVA dataset
    '''
    # initialize the scalar used in train process, 
    # AverageMeter is a class for keeping track of averages and value counts.
    batch_time = AverageMeter() # time to process each batch
    data_time = AverageMeter() # time to load data for each batch
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    end = time.time()

    # set model to eval mode
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    # remove the previous the result file and ground truth file
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print("remove {}".format(tmp_dir))
        print("all tmp files removed")

    #  iterate the validate set to train the model
    for idx, data in enumerate(data_loader):
        #update the data load time
        data_time.update(time.time() - end)

        # set device to GPU
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)

        # get video information from dataloader
        samples = data[0]

        # get targrt from the dataloader
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)

        # move sample and target to GPU
        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # get output from the model
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                outputs = model(samples)

        # calculate the each part of loss function
        loss_dict = criterion(outputs, targets)

        # update the weight of each part in loss function       
        weight_dict = criterion.weight_dict

        # pass output into post process. transfer the format of output and normalize the output
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)

        # save the ground truth and prediction value into specific format
        for bidx in range(scores.shape[0]):
            frame_id = batch_id[bidx][0]
            key_pos = batch_id[bidx][1]

            # save the prediction 
            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                out_key_pos = key_pos // cfg.CONFIG.MODEL.DS_RATE

                buff_output.append(scores[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_anno.append(boxes[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
                buff_binary.append(output_b[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
            else:
                buff_output.append(scores[bidx])
                buff_anno.append(boxes[bidx])
                buff_binary.append(output_b[bidx])

            for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                buff_id.extend([frame_id])

            raw_idx = (targets[bidx]["raw_boxes"][:, 1] == key_pos).nonzero().squeeze()

            # save the ground truth
            val_label = targets[bidx]["labels"][raw_idx]
            val_label = val_label.reshape(-1, val_label.shape[-1])
            raw_boxes = targets[bidx]["raw_boxes"][raw_idx]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())

            img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                           range(len(raw_boxes))]

            buff_GT_id.extend(img_id_item)

        batch_time.update(time.time() - end)
        end = time.time()

        # print the training detail of this epoch
        if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)

            # reduce loss value on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            # update the loss value
            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))

            # if loss is infinite, stop evaluating
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping eval".format(loss_value))
                print(loss_dict_reduced)
                exit(1)
            # print the loss at this epoch
            print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}, loss_ce_b: {loss_ce_b:.3f}'.format(
                class_error=class_err.avg,
                loss=losses_avg.avg,
                loss_bbox=losses_box.avg,
                loss_giou=losses_giou.avg,
                loss_ce=losses_ce.avg,
                loss_ce_b=losses_ce_b.avg,
            )
            print(print_string)

    # write loss information into tensorboard to visualization the trainning process
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        writer.add_scalar('val/class_error', class_err.avg, epoch)
        writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
        writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
        writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
        writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)
        writer.add_scalar('val/loss_ce_b', losses_ce_b.avg, epoch)

    # save the ground truth and prediciton into log
    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    buff_binary = np.concatenate(buff_binary, axis=0)
    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    print(buff_output.shape, buff_anno.shape, buff_binary.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))

    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x], buff_binary[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    # write files and align all workers
    # torch.distributed.barrier()

    # evaluate the result
    Map_ = 0
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # initial the evaluater
        evaluater = STDetectionEvaluater(cfg.CONFIG.DATA.LABEL_PATH, class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        # read ground truth and result then pass into evaluater
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)

        # evaluate the result
        mAP, metrics = evaluater.evaluate()

        # print the mAP and AP of each category
        print(metrics)
        print_string = 'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]

        # evaluate the precision of the  bounding box detection
        evaluater = STDetectionEvaluaterSinglePerson(cfg.CONFIG.DATA.LABEL_PATH)
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = 'person AP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        writer.add_scalar('val/val_person_AP_epoch', mAP[0], epoch)
    # torch.distributed.barrier()
    time.sleep(30)
    return Map_

@torch.no_grad()
def validate_tuber_jhmdb_detection(cfg, model, criterion, postprocessors, data_loader, epoch, writer):
    '''
        this function is used to evaluate the JHMDB dataset
    '''
    
    # initialize the scalar used in train process
    # AverageMeter is a class for keeping track of averages and value counts.
    batch_time = AverageMeter() # time to process each batch
    data_time = AverageMeter()  # time to load data for each batch
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()

    # set model to eval mode
    end = time.time()
    model.eval()
    criterion.eval()

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_binary = []

    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []
    
    # remove the previous the result file and ground truth file
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tmp_path = "{}/{}".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR)
        if not os.path.exists(tmp_path): os.makedirs(tmp_path)
        tmp_dirs_ = glob.glob("{}/{}/*.txt".format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR))
        for tmp_dir in tmp_dirs_:
            os.remove(tmp_dir)
            print("remove {}".format(tmp_dir))
        print("all tmp files removed")

    #  iterate the validate set to train the model
    for idx, data in enumerate(data_loader):
        #update the data load time
        data_time.update(time.time() - end)

        # set device to GPU
        device = "cuda:" + str(cfg.DDP_CONFIG.GPU)

        # get video information from dataloader
        samples = data[0]

        # get targrt from the dataloader
        if cfg.CONFIG.TWO_STREAM:
            samples2 = data[1]
            targets = data[2]
            samples2 = samples2.to(device)
        else:
            targets = data[1]

        if cfg.CONFIG.USE_LFB:
            if cfg.CONFIG.USE_LOCATION:
                lfb_features = data[-2]
                lfb_features = lfb_features.to(device)

                lfb_location_features = data[-1]
                lfb_location_features = lfb_location_features.to(device)
            else:
                lfb_features = data[-1]
                lfb_features = lfb_features.to(device)
        # move sample and target to GPU
        samples = samples.to(device)
        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]      

        # get output from the model
        if cfg.CONFIG.TWO_STREAM:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, samples2, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, samples2, lfb_features)
            else:
                outputs = model(samples, samples2)
        else:
            if cfg.CONFIG.USE_LFB:
                if cfg.CONFIG.USE_LOCATION:
                    outputs = model(samples, lfb_features, lfb_location_features)
                else:
                    outputs = model(samples, lfb_features)
            else:
                outputs = model(samples)

        # calculate the each part of loss function
        loss_dict = criterion(outputs, targets)

        # update the weight of each part in loss function       
        weight_dict = criterion.weight_dict

        # pass output into post process. transfer the format of output and normalize the output
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)

        # save the ground truth and prediction value into specific format
        for bidx in range(scores.shape[0]):
            
            # save the prediction 
            if len(targets[bidx]["raw_boxes"]) == 0:
                continue

            frame_id = batch_id[bidx][0]
            key_pos = batch_id[bidx][1]
            out_key_pos = key_pos

            buff_output.append(scores[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
            buff_anno.append(boxes[bidx, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])

            for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                buff_id.extend([frame_id])
                buff_binary.append(output_b[..., 0])

            # save the ground truth
            val_label = targets[bidx]["labels"]
            val_category = torch.full((len(val_label), 21), 0)
            for vl in range(len(val_label)):
                label = int(val_label[vl])
                val_category[vl, label] = 1
            val_label = val_category

            raw_boxes = targets[bidx]["raw_boxes"]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())

            img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                           range(len(raw_boxes))]

            buff_GT_id.extend(img_id_item)

        # update the batch time
        batch_time.update(time.time() - end)
        end = time.time()

        # print the training detail of this epoch
        if (cfg.DDP_CONFIG.GPU_WORLD_RANK == 0):
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)


            # reduce loss value on single GPU           
            loss_dict_reduced = loss_dict
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            # update the loss value
            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            # if loss is infinite, stop evaluating
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping eval".format(loss_value))
                print(loss_dict_reduced)
                exit(1)

            # print the loss at this epoch
            print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}'.format(
                class_error=class_err.avg,
                loss=losses_avg.avg,
                loss_bbox=losses_box.avg,
                loss_giou=losses_giou.avg,
                loss_ce=losses_ce.avg
            )
            print(print_string)

    # write loss information into tensorboard to visualization the trainning process
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        writer.add_scalar('val/class_error', class_err.avg, epoch)
        writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
        writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
        writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
        writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)


    # save the ground truth and prediciton into log
    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    buff_binary = np.concatenate(buff_binary, axis=0)

    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)

    print(buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))

    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))

    tmp_binary_path = '{}/{}/binary_{}.txt'
    with open(tmp_binary_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = buff_binary[x]
            f.write("{} {}\n".format(buff_id[x], data.tolist()))

    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    # write files and align all workers
    # torch.distributed.barrier()

    # evaluate the result
    Map_ = 0
    # aggregate files
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # read results
        # initial the evaluater
        evaluater = STDetectionEvaluaterJHMDB(class_num=cfg.CONFIG.DATA.NUM_CLASSES)

        # read ground truth and result then pass into evaluater
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(cfg.DDP_CONFIG.GPU_WORLD_SIZE)]
        evaluater.load_detection_from_path(file_path_lst)

        # evaluate the result
        mAP, metrics = evaluater.evaluate()

        # print the mAP and AP of each category
        print(metrics)
        print_string = 'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]
    # torch.distributed.barrier()
    return Map_

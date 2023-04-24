import pandas as pd
import cv2
import torch.utils.data as data
from glob import glob
import numpy as np
from utils.misc import collate_fn
import torch
import random
from PIL import Image
import torch.nn.functional as F
import datasets.video_transforms as T
import json


class VideoDataset(data.Dataset):
    '''
        this class extend pytorch class data, to read, pre-process and store the video data
        this class implement by rewriting __getitem__() and __len__()

        Parameter:
            frame_path : path of dataset
            video_frame_bbox : bounding box annotation
            frame_keys_list : list of the frame name
            clip_len : clip length of a sample
            frame_sample_rate :  rate to sample frame
            transforms : composed function to transform the input frame
            crop_size : crop size for transform input frame
            resize_size : resize size for transform input frame
            mode : type of the dataset, train set or validation set.
            class_num: number of the action categories
        Attributes:
            __getitem__(): get video and annotation, called by dataloader
            __len__() : get the length of the dataset
            load_annotation() : load annotation for given frame
            loadvideo() : load video clip for given frame
    '''

    def __init__(self, frame_path, video_frame_bbox, frame_keys_list, clip_len, frame_sample_rate,
                 transforms, crop_size=224, resize_size=256, mode="train", class_num=80):
        
        self.video_frame_bbox = video_frame_bbox
        self.video_frame_list = frame_keys_list 
        self.frame_path = frame_path 

        self.crop_size = crop_size 
        self.clip_len = clip_len 
        self.frame_sample_rate = frame_sample_rate
        self.class_num = class_num  
        self.resize_size = resize_size 

        self.index_cnt = 0
        self._transforms = transforms
        self.mode = mode

        print("rescale size: {}, crop size: {}".format(resize_size, crop_size))

    def __getitem__(self, index):

        # get the name of frame
        frame_key = self.video_frame_list[index] 

        # split the frame name, former is the video name， latter is the timestamp of the frame
        vid, frame_second = frame_key.split(",")

        # time stamp of AVA is 902 - 1798, reset the timestamp
        timef = int(frame_second) - 900

        # set the start of image time stamp and load video clip for given frame
        start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))
        imgs, target = self.loadvideo(start_img, vid, frame_key)

        # transform the input frame
        if len(target) == 0 or target['boxes'].shape[0] == 0:
            pass
        else:
            if self._transforms is not None:
                imgs, target = self._transforms(imgs, target)

        # resample, if there is no bounding box in given frame
        # random choose a frame from the dataset and sample again
        while len(target) == 0 or target['boxes'].shape[0] == 0:
            print('resample.')
            self.index_cnt -= 1
            index = np.random.randint(len(self.video_frame_list))
            frame_key = self.video_frame_list[index]
            vid, frame_second = frame_key.split(",")
            timef = int(frame_second) - 900

            start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

            imgs, target = self.loadvideo(start_img, vid, frame_key)

            if len(target)==0 or target['boxes'].shape[0] == 0:
                pass
            else:
                if self._transforms is not None:
                    imgs, target = self._transforms(imgs, target)

        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.permute(1, 0, 2, 3)

        return imgs, target

    def load_annotation(self, sample_id, video_frame_list):
        '''
            the function to get the annotation of the given frame and fit in resized image

            Parameters:
                sample_id: key of the frame
                video_frame_list: list of the frame address of given frame's video 
        '''
        
        num_classes = self.class_num
        boxes, classes = [], []
        target = {}

        # get the resolution of the origin video
        first_img = cv2.imread(video_frame_list[0])

        oh = first_img.shape[0]
        ow = first_img.shape[1]

        # calculate the image size after resize
        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)

        p_t = int(self.clip_len // 2)
        key_pos = p_t

        # get the original annotation
        anno_entity = self.video_frame_bbox[sample_id]

        # transform the annotation to resized size
        for i, bbox in enumerate(anno_entity["bboxes"]):
            label_tmp = np.zeros((num_classes, ))
            acts_p = anno_entity["acts"][i]
            for l in acts_p:
                label_tmp[l] = 1

            if np.sum(label_tmp) == 0: continue
            p_x = np.int(bbox[0] * nw)
            p_y = np.int(bbox[1] * nh)
            p_w = np.int(bbox[2] * nw)
            p_h = np.int(bbox[3] * nh)

            boxes.append([p_t, p_x, p_y, p_w, p_h])
            classes.append(label_tmp)

        # transform the annotation to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        boxes[:, 1::3].clamp_(min=0, max=int(nw))
        boxes[:, 2::3].clamp_(min=0, max=nh)

        if boxes.shape[0]:
            raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt)
        else:
            raw_boxes = boxes

        classes = torch.as_tensor(classes, dtype=torch.float32).reshape(-1, num_classes)

        # save and return the annotation
        target["image_id"] = [str(sample_id).replace(",", "_"), key_pos]
        target['boxes'] = boxes
        target['raw_boxes'] = raw_boxes
        target["labels"] = classes
        target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
        target["size"] = torch.as_tensor([int(nh), int(nw)])
        self.index_cnt = self.index_cnt + 1

        return target

    def loadvideo(self, start_img, vid, frame_key):
        '''
            function to read frame and corresponding annotation

            Parameters: 
                start_img: timestamp of the start image of the video clip
                vid : video ID of the given frame
                frame_key : key of the given frame
        '''
        # get the list of all frame for the given frame's video
        video_frame_path = self.frame_path
        video_frame_list = sorted(glob(video_frame_path.format(vid)+ '/*.jpg'))

        # judge whether the path is valid
        if len(video_frame_list) == 0:
            print("path doesnt exist", video_frame_path)
            return [], []

        # get annotation for the given frame
        target = self.load_annotation(frame_key, video_frame_list)

        # read the video clip from start iamge frame by frame. then store in buffer
        start_img = np.max(start_img, 0)
        end_img = start_img + self.clip_len * self.frame_sample_rate
        indx_img = list(np.clip(range(start_img, end_img, self.frame_sample_rate), 0, len(video_frame_list) - 1))
        buffer = []
        for frame_idx in indx_img:
            tmp = Image.open(video_frame_list[frame_idx])
            tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
            buffer.append(tmp)

        return buffer, target

    def __len__(self):
        return len(self.video_frame_list)


def make_transforms(image_set, cfg):
    '''
        choose the transform funtion for different type task
    '''
    # normalization funtion, transform input into tensor and normalization with zero-score scaling
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("transform image crop: {}".format(cfg.CONFIG.DATA.IMG_SIZE))
    # different composed transform function for train set and validation set
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            T.ColorJitter(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataloader(cfg):
    '''
        build dataloader for train set and validate set which are ready to feed in model
        dataloader will pass the sample in VideoDataset into model iteratively
    '''
    #  from annotion file get annotation information for train set
    train_bbox_json = json.load(open(cfg.CONFIG.DATA.ANNO_PATH.format("train")))
    train_video_frame_bbox, train_frame_keys_list = train_bbox_json["video_frame_bbox"], train_bbox_json["frame_keys_list"]

    # instance VideoDataset for train set
    train_dataset = VideoDataset(cfg.CONFIG.DATA.DATA_PATH,
                                 train_video_frame_bbox,
                                 train_frame_keys_list,
                                 transforms=make_transforms("train", cfg),
                                 frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                 clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                                 resize_size=cfg.CONFIG.DATA.IMG_RESHAPE_SIZE,
                                 crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                                 mode="train")
    
    #  from annotion file get annotation information for train set
    val_bbox_json = json.load(open(cfg.CONFIG.DATA.ANNO_PATH.format("val")))
    val_video_frame_bbox, val_frame_keys_list = val_bbox_json["video_frame_bbox"], val_bbox_json["frame_keys_list"]


    # instance VideoDataset for train set
    val_dataset = VideoDataset(cfg.CONFIG.DATA.DATA_PATH,
                               val_video_frame_bbox,
                               val_frame_keys_list,
                               transforms=make_transforms("val", cfg),
                               frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                               clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                               resize_size=cfg.CONFIG.DATA.IMG_SIZE,
                               crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                               mode="val")

    if cfg.DDP_CONFIG.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, cfg.CONFIG.TRAIN.BATCH_SIZE, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        batch_sampler_train = None

    # build dataloader for both train set and validation set
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=(train_sampler is None),
                                               num_workers=9, pin_memory=True, batch_sampler=batch_sampler_train,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=9, sampler=val_sampler, pin_memory=True, collate_fn=collate_fn)

    print(cfg.CONFIG.DATA.ANNO_PATH.format("train"), cfg.CONFIG.DATA.ANNO_PATH.format("val"))

    return train_loader, val_loader, train_sampler, val_sampler, None


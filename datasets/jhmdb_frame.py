import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.utils.data
import torch.nn.functional as F
import datasets.video_transforms as T
from utils.misc import collate_fn
from glob import glob



class VideoDataset(Dataset):
    '''
        this class extend pytorch class data, to read, pre-process and store the video data
        this class implement by rewriting __getitem__() and __len__()

        Parameter:
            directory : path of annotation file
            video_path : path of the video
            transforms : composed function to transform the input frame

            clip_len : clip length of a sample
            crop_size : crop size for transform input frame
            resize_size : resize size for transform input frame
            mode : type of the dataset, train set or validation set.
        Attributes:
            __getitem__(): get video and annotation, called by dataloader
            __len__() : get the length of the dataset
            load_annotation() : load annotation for given frame
            loadvideo() : load video clip for given frame
    '''

    def __init__(self, directory, video_path, transforms, clip_len=8, crop_size=224, resize_size=256,
                 mode='train'):
        
        # read the annotation information from a pickle file
        self.directory = directory
        cache_file = os.path.join(directory, 'JHMDB-GT.pkl')
        assert os.path.isfile(cache_file), "Missing cache file for dataset "
        with open(cache_file, 'rb') as fid:
            dataset = pickle.load(fid, encoding='iso-8859-1')


        self.video_path = video_path
        self._transforms = transforms
        self.dataset = dataset
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.index_cnt = 0

        # get a list of keyframes
        self.index_to_sample_t = []

        if mode == 'val' or mode == 'test':
            self.dataset_samples = self.dataset['test_videos'][0]
        elif mode == 'train':
            self.dataset_samples = self.dataset['train_videos'][0]

        if mode == 'test':
            for vid in self.dataset_samples:
                self.index_to_sample_t += [(vid, i) for i in range(self.dataset['nframes'][vid])]
        else:
            for vid in self.dataset_samples:
                self.index_to_sample_t += [(vid, i) for i in range(self.dataset['nframes'][vid])]

        # self.index_to_sample_t = self.index_to_sample_t[:200]

        print(self.index_to_sample_t.__len__(), "frames indexed")

        self.labelmap = self.dataset['labels']
        self.max_person = 0
        self.person_size = 0

    def __getitem__(self, index):
        # get the video name and frame id 
        sample_id, frame_id = self.index_to_sample_t[index]
        p_t = self.clip_len // 2

        # load video clip and annotion for the given frame
        target = self.load_annotation(sample_id, frame_id, p_t)
        imgs = self.loadvideo(frame_id, sample_id, target, p_t)

        # apply tranform on input image sequence for data augmentation
        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target)
        if self.mode == 'test':
            if target['boxes'].shape[0] == 0:
                target['boxes'] = torch.concat([target["boxes"], torch.from_numpy(np.array([[0, 0, 0, 1, 1]]))])
                target['labels'] = torch.concat([target["labels"], torch.from_numpy(np.array([0]))])
                target['area'] = torch.concat([target["area"], torch.from_numpy(np.array([30]))])
                target['raw_boxes'] = torch.concat([target["raw_boxes"], torch.from_numpy(np.array([[0, 0, 0, 0, 1, 1]]))])

        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.permute(1, 0, 2, 3)

        return imgs, target

    def load_annotation(self, sample_id, frame_id, p_t):
        '''
            the function to get the annotation of the given frame and fit in resized image

            Parameters:
                sample_id: video ID
                frame_id: frame ID
                p_t: half length of the video clip
        '''

        boxes, classes = [], []
        target = {}
        vis = [0]

        # get the resolution of the origin video
        oh = self.dataset['resolution'][sample_id][0]
        ow = self.dataset['resolution'][sample_id][1]

        # calculate the image size after resize
        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)

        key_pos = p_t

        # transform the annotation to resized size
        for ilabel, tubes in self.dataset['gttubes'][sample_id].items():
            for t in tubes:
                box_ = t[(t[:, 0] == frame_id), 0:5]
                key_point = key_pos // 8

                if len(box_) > 0:
                    box = box_[0]
                    p_x1 = np.int(box[1] / ow * nw)
                    p_y1 = np.int(box[2] / oh * nh)
                    p_x2 = np.int(box[3] / ow * nw)
                    p_y2 = np.int(box[4] / oh * nh)
                    boxes.append([key_pos, p_x1, p_y1, p_x2, p_y2])
                    classes.append(np.clip(ilabel, 0, 24))

                    vis[0] = 1

        if self.mode == 'test' and False:
            classes = torch.as_tensor(classes, dtype=torch.int64)
            # print('classes', classes.shape)

            target["image_id"] = [str(sample_id) + '-' + str(frame_id)]
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            self.index_cnt = self.index_cnt + 1

        else:
            # transform the annotation to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
            boxes[:, 1::3].clamp_(min=0, max=nw)
            boxes[:, 2::3].clamp_(min=0, max=nh)

            if boxes.shape[0]:
                raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt)
            else:
                raw_boxes = boxes

            classes = torch.as_tensor(classes, dtype=torch.int64)

            # save and return the annotation
            target["image_id"] = [str(sample_id).replace("/", "_") + '-' + str(frame_id), key_pos]
            target["key_pos"] = torch.as_tensor(key_pos)
            target['boxes'] = boxes
            target['raw_boxes'] = raw_boxes
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            target["vis"] = torch.as_tensor(vis)
            self.index_cnt = self.index_cnt + 1
        return target

    def loadvideo(self, mid_point, sample_id, target, p_t):
        '''
            function to load the video based on keyframe

            Parameters: 
                mid_point : id of the target frame
                sample_id : video ID which include the target frame
                target: annotation of the target frame
                p_t : half length of the clip length
        '''
        from PIL import Image
        import numpy as np

        # read the video clip from start iamge frame by frame. then store in buffer
        buffer = []
        start = max(mid_point - p_t, 0)
        end = min(mid_point + self.clip_len - p_t, self.dataset["nframes"][sample_id] - 1)
        frame_ids_ = [s for s in range(start, end)]
        # fill blank frames with the first and last frames of the video clip
        if len(frame_ids_) < self.clip_len:
            front_size = (self.clip_len - len(frame_ids_)) // 2
            front = [0 for _ in range(front_size)]
            back = [end for _ in range(self.clip_len - len(frame_ids_) - front_size)]
            frame_ids_ = front + frame_ids_ + back
        assert len(frame_ids_) == self.clip_len

        # read in the frame
        for frame_idx in frame_ids_:
            tmp = Image.open(os.path.join(self.video_path, sample_id, "{:0>5}.png".format(frame_idx + 1)))
            try:
                tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
            except:
                print(target)
                raise "error"
            buffer.append(np.array(tmp))
        buffer = np.stack(buffer, axis=0)

        imgs = []
        for i in range(buffer.shape[0]):
            imgs.append(Image.fromarray(buffer[i, :, :, :].astype(np.uint8)))
        return imgs

    def __len__(self):
        return len(self.index_to_sample_t)


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
            # T.HorizontalFlip(),
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def build_dataloader(cfg):
    '''
        build dataloader for train set and validate set which are ready to feed in model
        dataloader will pass the sample in VideoDataset into model iteratively
    '''

    # instance VideoDataset for train set and validation set

    train_dataset = VideoDataset(directory=cfg.CONFIG.DATA.ANNO_PATH,
                                 video_path=cfg.CONFIG.DATA.DATA_PATH,
                                 transforms=make_transforms("train", cfg),
                                 clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                                 resize_size=cfg.CONFIG.DATA.IMG_RESHAPE_SIZE,
                                 crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                                 mode="train")

    val_dataset = VideoDataset(directory=cfg.CONFIG.DATA.ANNO_PATH,
                               video_path=cfg.CONFIG.DATA.DATA_PATH,
                               transforms=make_transforms("val", cfg),
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

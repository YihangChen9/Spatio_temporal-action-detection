# -*- coding:utf8 -*-
import cv2
import os
import pickle
import numpy as np
import shutil
import time
import torch
import torch.optim
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import torch.nn.functional as F
import datasets.video_transforms as T
from utils.misc import collate_fn
from glob import glob
from pipelines.launch import spawn_workers
from pipelines.video_action_recognition_config import get_cfg_defaults
from models.tuber import build_model
from utils.model_utils import deploy_model_jhmdb, load_model
from visualization_jhmdb import load_detection_from_path, parse_id,makedir,picvideo

def get_frame_from_video(video_name, frame_per_sec):
    """
    Parameters:
        video_name: path of the video
        frame_per_sec: number of frame extract from 1 sec
    Returns:
        total number of frame
    """
 
    # path to save the image
    save_path = 'video/JHMDB/test_frames/test/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)
 
    # read the video
    video_capture = cv2.VideoCapture(video_name)
    fps = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
    print(fps)
    if (fps < frame_per_sec):
        print("fps of this video is too low")
        exit(-1)

        return
    interval = fps//frame_per_sec
    i = 0
    j = 0
 
    while True:
        success, frame = video_capture.read()
        i += 1
        if not success:
            print('video is all read')
            break
        if i % interval == 0:
            # save the image
            j += 1
            save_name = str(j).zfill(5)+ '.png'
            cv2.imwrite(save_path + save_name, frame)
            print('image of %s is saved' % save_name)
    return j


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

    def __init__(self, directory, video_path, transforms,frame, clip_len=8, crop_size=224, resize_size=256,
                 mode='test'):
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
        self.frame = frame

        # get a list of keyframes
        self.index_to_sample_t = []

        self.dataset_samples = ['test']
        vid = 'test'
        self.index_to_sample_t += [(vid, i) for i in range(self.frame)]

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
        target = self.load_annotation('wave/wave_and_say_hi_wave_u_nm_np1_fr_med_0', 1, 0, p_t)
        imgs = self.loadvideo(frame_id, sample_id, target, p_t)

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

    def load_annotation(self, sample_id, start, index, p_t):
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
                box_ = t[(t[:, 0] == start), 0:5]
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

            target["image_id"] = [str(sample_id) + '-' + str(start)]
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
            target["image_id"] = [str(sample_id).replace("/", "_") + '-' + str(start), key_pos]
            target["key_pos"] = torch.as_tensor(key_pos)
            target['boxes'] = boxes
            target['raw_boxes'] = raw_boxes
            target["labels"] = classes
            target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
            target["size"] = torch.as_tensor([int(nh), int(nw)])
            target["vis"] = torch.as_tensor(vis)
            self.index_cnt = self.index_cnt + 1
        return target

    # load the video based on keyframe
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
        end = min(mid_point + self.clip_len - p_t, self.frame - 1)
        frame_ids_ = [s for s in range(start, end)]
        # fill blank frames with the first and last frames of the video clip
        if len(frame_ids_) < self.clip_len:
            front_size = (self.clip_len - len(frame_ids_)) // 2
            front = [0 for _ in range(front_size)]
            back = [end for _ in range(self.clip_len - len(frame_ids_) - front_size)]
            frame_ids_ = front + frame_ids_ + back
        assert len(frame_ids_) == self.clip_len
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
 
 
def build_dataloader(cfg,frame):
    '''
        build dataloader for train set and validate set which are ready to feed in model
        dataloader will pass the sample in VideoDataset into model iteratively
    '''
    # instance VideoDataset for test set
    test_dataset = VideoDataset(directory=cfg.CONFIG.DATA.ANNO_PATH,
                               video_path=cfg.CONFIG.DATA.DATA_PATH,
                               transforms=make_transforms("val", cfg),
                               frame=frame,
                               clip_len=cfg.CONFIG.DATA.TEMP_LEN,
                               resize_size=cfg.CONFIG.DATA.IMG_SIZE,
                               crop_size=cfg.CONFIG.DATA.IMG_SIZE,
                               mode="test")

    val_sampler = None

    # build dataloader for test set
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=2, sampler=val_sampler, pin_memory=True, collate_fn=collate_fn)

    print(cfg.CONFIG.DATA.ANNO_PATH.format("train"), cfg.CONFIG.DATA.ANNO_PATH.format("val"))

    return  test_loader


def get_gt_visualization(detection_path):

    path1 = 'video/JHMDB/test_frames/'
    tempNum = ''
    tempOutPath = ''

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0) 
    color_de = (0,255,0)
    thickness = 2
    label_dict = parse_id()

    # get ground truth and result
    detection_bbox,detection_label,detection_scores = load_detection_from_path(detection_path)

    count = 1
    for key in detection_bbox.keys():
        num = str(count).zfill(5)
        count = count + 1
        path2 = path1 + 'test/' + num +'.png'
        # read the image
        image = cv2.imread(path2) 
        print(detection_bbox[key])        
        # same as the ground truth, draw the bounding box, label and classification socre of ground truth
        for i in range(len(detection_bbox[key])):       
            bbox_detected = detection_bbox[key][i].tolist()
            x1_de = int( float(bbox_detected[0]))
            y1_de = int( float(bbox_detected[1]))
        
            x2_de = int( float(bbox_detected[2]))
            y2_de = int( float(bbox_detected[3]))
            start_point_de = (x1_de,y1_de)
            end_point_de = (x2_de,y2_de)
            image = cv2.rectangle(image, start_point_de, end_point_de, color_de, thickness) 

            for i in range(len(detection_label[key])):
                action_id_de = detection_label[key][i]
                act_de=str(label_dict[action_id_de-1])
                prob = str(round(detection_scores[key][i]*100,2))
                text = act_de + "-"+prob+"%"
                image = cv2.putText(image, text, (x1_de, y1_de+10*(i+1)), font, 0.3, (0, 255, 255), 1)

        # save all the annotated frame
        tempOutPath = 'video/JHMDB/anno_test_frames/' + num + '.png'
        makedir(tempOutPath)
        cv2.imwrite(tempOutPath, image)

def picvideo(in_path,out_path,type):
    '''
        generate video from annotated video
    '''
    filelist = os.listdir(in_path)  # get the file in the path
    filelist.sort(reverse = False)  ##sort the frame
    '''
    fps:
    Frame rate: n images are written in 1 second [control an image to stay for 5 seconds, that is, the frame rate is 1, and repeat this image 5 times] 
    If you have 50 534*300 images in your folder, and you set 5 to play every second, the video will be 10 seconds long
    '''
    
    image = cv2.imread(in_path + '00001.png') 
    
    sp = image.shape
    
    size = (sp[1],sp[0])
    if (type.upper() == 'AVI'):
        video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('X','V','I','D'), 10, size)
    else:
        video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('X','2','6','4'), 10, size)

    for item in filelist:
        if item.endswith('.png'):  # judge whether the image is .png
            item = in_path  + item
            img = cv2.imread(item)  # use opencv read image，return numpy.array
            video.write(img)  # write image into video

    video.release()  

def main_worker(cfg):
    video_name = cfg.CONFIG.DATA.TEST_PATH
    frame_per_sec = 10
    frame = get_frame_from_video(video_name, frame_per_sec)
    # create model
    print('Creating TubeR model: %s' % cfg.CONFIG.MODEL.NAME)
    model, _, postprocessors = build_model(cfg)
    model = deploy_model_jhmdb(model, cfg, is_tuber=True)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create dataset and dataloader
    test_loader = build_dataloader(cfg,frame)

    print("test sampler", len(test_loader))

    # docs: add resume option
    model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)

    print('Start training...')
    start_time = time.time()
    max_accuracy = 0.0
    device = "cuda:" + str(cfg.DDP_CONFIG.GPU)
    buff_output = []
    buff_anno = []
    buff_id = []
    for idx, data in enumerate(test_loader):
        samples = data[0]
        samples = samples.to(device)
        targets = data[1]
        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        scores, boxes, output_b = postprocessors['bbox'](outputs, orig_target_sizes)
        out_key_pos = 16
        buff_output.append(scores[0, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
        buff_anno.append(boxes[0, out_key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(out_key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
        for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
            frame_id = '_test_'+str(idx)
            buff_id.extend([frame_id])

    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    tmp_path = 'log/test/test_result.txt'
    os.remove(tmp_path)
    with open(tmp_path, 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    print("eval finished")
    detection_path = [tmp_path]
    get_gt_visualization(detection_path)
    input_path = 'video/JHMDB/anno_test_frames/'
    output_path = 'video/JHMDB/anno_test_video/test.avi'
    type = cfg.CONFIG.DATA.TEST_PATH.rsplit(".",1)[-1]
    picvideo(input_path,output_path,type)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='configuration/Tuber_CSN152_JHMDB.yaml',
                        help='path to config file.')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)

    


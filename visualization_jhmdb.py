import cv2  
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import time
import os


def makedir(dir_path):
    '''
        make directory if the directory is not exisit
    '''
    dir_path=os.path.dirname(dir_path)
    bool=os.path.exists(dir_path)
    if bool:
        pass
    else:
        os.makedirs(dir_path)


def parse_id():
    '''
        get the categories of JHMDB
    '''
    activity_list = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
    categories = []
    for i, act_name in enumerate(activity_list):
        categories.append({i + 1: act_name})
    return categories


def split_key(image_key):
    '''
        split the frame_key of JHMDB dataset
    '''
    one_word = ['catch','clap','golf','jump','pick','pour','pullup','push','run','sit','stand','throw','walk','wave']
    two_word = ['brush_hair','climb_stairs','kick_ball','shoot_ball','shoot_bow','shoot_gun','swing_baseball']

    splited_word = image_key.split("_")
    action = ''
    frame = ''
    if splited_word[0] in one_word:
        action, frame = image_key.split("_",1)
    else:
        first,second,third = image_key.split("_",2)

        action = first + "_" +second
        frame = third
    return action, frame
    



def load_GT_from_path(file_lst):
    # loading data from files
    t_end = time.time()
    GT = {}
    for path in file_lst:
        data = open(path).readlines()
        for line in data:
            # sparse the frame key
            image_key = line.split(' [')[0]
            data = line.split(' [')[1].split(']')[0].split(',')
            data = [float(x) for x in data]
            # if (data[4] - data[2]) * (data[5] - data[3]) < 10:
            #     self.exclude_key.append(image_key)
            #     continue
            scores = np.array(data[6:])
            if not image_key in GT:
                GT[image_key] = {
                    'bbox': [],
                    'labels': [],
                    'scores': [],
                }

            # save ground truth 
            for x in range(len(scores)):
                if scores[x] <= 1e-2: continue
                GT[image_key]['bbox'].append(
                    np.asarray([data[2], data[3], data[4], data[5]], dtype=float)
                )
                GT[image_key]['labels'].append(x + 1)
                GT[image_key]['scores'].append(scores[x])
    return GT
    
def load_detection_from_path(file_lst):
    # loading data from files
    t_end = time.time()
    detection = {}

    n = 0
    for path in file_lst:
        data = open(path).readlines()
        for line in data:
            # sparse the frame key
            image_key = line.split(' [')[0]
            data = line.split(' [')[1].split(']')[0].split(',')
            data = [float(x) for x in data]
            scores = np.array(data[4:21 + 4])
            if np.argmax(np.array(data[4:])) == len(np.array(data[4:])) - 1:
                continue
                # scores = np.array(data[4:self.class_num + 4])

            if not image_key in detection:
                detection[image_key] = {
                    'bbox': [],
                    'labels': [],
                    'scores': [],
                }

            x = np.argmax(scores)
            max_index = scores.argsort()[-3:][::-1]

            # get the top 3 best label, score should > 0.01
            for index in max_index:
                if scores[index] < 0.01:
                    continue
                else:
                    detection[image_key]['labels'].append(index+1)
                    detection[image_key]['scores'].append(scores[index])
            # if scores[x] <= 1e-1: continue
            detection[image_key]['bbox'].append(
                np.asarray([data[0], data[1], data[2], data[3]], dtype=float)
            )
    count = 0
    detection_bbox={}
    detection_label={}
    detection_scores={}
    for image_key, info in detection.items():

        if len(info['bbox']) == 0:
            print(count)
            continue
        #sorted by confidence:
        boxes = np.vstack(info['bbox'])
        labels = np.array(info['labels'], dtype=int)
        scores = np.array(info['scores'], dtype=float)

        # save the result
        detection_bbox[image_key] = boxes
        detection_label[image_key] = labels
        detection_scores[image_key] = scores
        count += 1

    return detection_bbox,detection_label,detection_scores




def get_gt_visualization(gt_path,detection_path):

    path1 = 'video/JHMDB/Frames/'
    tempNum = ''
    tempOutPath = ''

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0) 
    color_de = (0,255,0)
    thickness = 2
    label_dict = parse_id()

    # get ground truth and result
    gt_data = load_GT_from_path(gt_path)
    detection_bbox,detection_label,detection_scores = load_detection_from_path(detection_path)

    for key in gt_data:

        action, frame_id = split_key(key)
        vid, fid = frame_id.rsplit("-",1)
        num = fid.zfill(5)
        path2 = path1 + action + '/' + vid +'/' + num + '.png'
        
        # read the image
        image = cv2.imread(path2) 
        bbox = gt_data[key]['bbox'][0]

        sp = image.shape
        h = sp[0]
        w = sp[1]
        x1 = int( float(bbox[0])  )
        y1 = int( float(bbox[1])  )
        
        x2 = int( float(bbox[2])  )
        y2 = int( float(bbox[3])  )
    
        # draw the bounding box, label and classification socre of ground truth
        start_point = (x1,y1)
        end_point = (x2,y2)
        image = cv2.rectangle(image, start_point, end_point, color, thickness) 
        act = ''
        action_id = gt_data[key]['labels'][0]
        act=str(label_dict[action_id-1])
        image = cv2.putText(image, act, (x2, y2), font, 0.3, (255, 0, 255), 1)

        # same as the ground truth, draw the bounding box, label and classification socre of ground truth
        if key in detection_bbox.keys():
            for i in range(len(detection_bbox[key])):       
                bbox_detected = detection_bbox[key][i].tolist()
                x1_de = int( float(bbox_detected[0])  )
                y1_de = int( float(bbox_detected[1])  )
            
                x2_de = int( float(bbox_detected[2])  )
                y2_de = int( float(bbox_detected[3])  )
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
        tempOutPath = 'video/JHMDB/anno_frames/' + action + '/' + vid + '/' + num + '.png'
        makedir(tempOutPath)
        cv2.imwrite(tempOutPath, image)

    


def picvideo(in_path,out_path):
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
    print(size)
    
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('X','2','6','4'), 10, size)

    for item in filelist:
        if item.endswith('.png'):  # judge whether the image is .png
            item = in_path  + item
            img = cv2.imread(item)  # use opencv read image，return numpy.array
            video.write(img)  # write image into video

    video.release()  

def gen_video():
    '''
        visulization all the sample video
    '''
    activity_list = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
    path1 = "video/JHMDB/anno_frames/"
    out_path1 = "video/JHMDB/anno_video/"
    for action in activity_list:
        path2 = path1 + action + "/"
        out_path2 = out_path1 + action +"/"
        makedir(out_path2)
        filelist = os.listdir(path2)
        for file in filelist:
            if file == '.DS_Store':
                continue
            pic_list = path2+file +'/'
            out_path3 = out_path2+file+"_anno.mp4"
            picvideo(pic_list,out_path3,file)
    return

if __name__ == '__main__':
    tmp_path = '{}/{}/{}.txt'
    tmp_GT_path = '{}/{}/GT_{}.txt'
    tmp_detection_path = '{}/{}/{}.txt'

    BASE_PATH = 'log/JHBDM_TUBER'
    RES_DIR='tmp_jhmdb'
    gt_path = [tmp_GT_path.format(BASE_PATH, RES_DIR, x) for x in range(7)]
    detection_path = [tmp_detection_path.format(BASE_PATH, RES_DIR, x) for x in range(7)]

    get_gt_visualization(gt_path,detection_path)
    gen_video()

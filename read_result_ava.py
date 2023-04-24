from evaluates.evaluate_ava import STDetectionEvaluater,STDetectionEvaluaterSinglePerson
from utils.print_ap import AP_to_csv

def read_result():
    tmp_path = '{}/{}/{}.txt'
    tmp_GT_path = '{}/{}/GT_{}.txt'
    label_path = 'datasets/assets/ava_action_list_v2.1_for_activitynet_2018.pbtxt'
    BASE_PATH = 'log/AVA_Tuber'
    RES_DIR='tmp2'

    # COCO, VOC, STRICT
    eval = 'VOC'

    # three different benchmark to evaluate performance of action detection
    if eval == 'COCO':
        iou_thresholds = [round(x/100, 2) for x in range(50, 96, 5)]
    elif eval =='VOC':
        iou_thresholds = [0.5]
    else:
        iou_thresholds = [0.75]

    # run evaluation method
    evaluater = STDetectionEvaluater(label_path,tiou_thresholds = iou_thresholds,class_num=80)
    file_path_lst = [tmp_GT_path.format(BASE_PATH, RES_DIR, x) for x in range(7)]
    evaluater.load_GT_from_path(file_path_lst)
    file_path_lst = [tmp_path.format(BASE_PATH, RES_DIR, x) for x in range(7)]
    evaluater.load_detection_from_path(file_path_lst)
    mAP, metrics = evaluater.evaluate()

    # print mAP
    if eval == 'COCO':
        coco_Map = sum(mAP) / len(mAP)
        print_string = eval + ' mAP: {mAP:.5f}'.format(mAP=coco_Map)
        print(print_string)
        print(mAP)

    else:
        print(metrics)
        print_string = eval +'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)

    # evaluater = STDetectionEvaluaterSinglePerson(label_path)
    # file_path_lst = [tmp_GT_path.format(BASE_PATH, RES_DIR, x) for x in range(7)]
    # evaluater.load_GT_from_path(file_path_lst)
    # file_path_lst = [tmp_path.format(BASE_PATH,RES_DIR, x) for x in range(7)]
    # evaluater.load_detection_from_path(file_path_lst)
    # mAP, metrics = evaluater.evaluate()
    # print(metrics)
    # print_string = 'person AP: {mAP:.5f}'.format(mAP=mAP[0])
    # print(print_string)

if __name__ == '__main__':
    out_path = 'result/AVA_AP.csv'

    metrics = read_result()
    AP_to_csv(metrics,out_path)

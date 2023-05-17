# Spatio-temporal Action Detection

This repo is the code of the Spatio-temporal action detecion project which implement the tubelet-level transformer-based spatio-temporal action detection model--[TubeR](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_TubeR_Tubelet_Transformer_for_Video_Action_Detection_CVPR_2022_paper.pdf).  
This project will train and evaluate TubeR models on AVA2.1 and JHMDB datasets.

## Results

| Dataset  |  Backbone | Pre-train  | #view  | IoU   | mAP  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| AVA2.1  | CSN-152  | Kinetics-400+IG65M  | 1 view  | 0.5  | 0.29  |
| JHMDB-21  |  CSN-152 |  Kinetics-400+IG65M | 1 view  |   0.5  | 0.72|


### Dependency
The project is tested working on:
- Python 3.7.12
- Torch 1.12.1(initial) and Torch 2.0.0(updated)
- CUDA 12.1
- timm==0.4.5 
- tensorboardX

### Dataset and pre-train model
To run the code of this project, you should download following file from github or given resource, since the code limit 250M on moodle.
Here is the github link:
https://github.com/YihangChen9/Spatio_temporal-action-detection

1. Please download the [asset.zip](https://drive.google.com/file/d/1u2Vr_PBVb3M-cN7pj-F-yMVAuMENhKl3/view?usp=share_link)(annotation of the AVA dataset) from the google drive. Then unzip into datasets/ directory
2. Please download the [pre-train.zip](https://drive.google.com/file/d/1xoFLJhxRlfLW9xNM1Txer7H3BNWgDYpu/view?usp=share_link)(trained model) directory from the the google drive. then put under tuber directory.    
   TubeR_CSN152_JHMDB.pth is the trained model for JHMDB  
   TubeR_CSN152_AVA.pth is the trained model for JHMDB  
   irCSN_152_ft_kinetics_from_ig65m_f126851907.mat is the pre-train model of CSN152 as backbone  
   detr.pth initial transformer parameter setting for AVA TubeR  

3. You can get [JHMDB](https://drive.google.com/file/d/1JFZomNYiTkfmjPX1M6syVAHCTm0jRtmj/view?usp=share_link) in this link, please download JHMDB.tar.gz. JHMDB-GT.pkl is the annotaiton file.
  

4. To get AVA dataset, please run the three bash scripts in the datasets directory one by one(please set the path in bash file):  
   download_ava.sh(down the original video clip)  
   chunk_video.sh(chunk the clip from 15 min to 30 min)  
   extract_frame.sh(extract frame from the clip)  



### Evaluation
To evaluate the model, first modify the config file(in configuration,one for AVA, another one for JHMDB):
- set the correct `WORLD_SIZE`, `GPU_WORLD_SIZE`, `DIST_URL`, `WOLRD_URLS` based on experiment setup.
- set the `LABEL_PATH`, `ANNO_PATH`, `DATA_PATH` to your local directory accordingly.
- Download the pre-trained model and set `PRETRAINED_PATH` to model path.
- make sure `LOAD` and `LOAD_FC` are set to True

Then set path to tuber and run:
```
# run evaluation
python3  eval_tuber_jhmdb.py
python3  eval_tuber_ava.py

```

### Training
To train TubeR from scratch, first modify the configfile:
- set the correct `WORLD_SIZE`, `GPU_WORLD_SIZE`, `DIST_URL`, `WOLRD_URLS` based on experiment setup.
- set the `LABEL_PATH`, `ANNO_PATH`, `DATA_PATH` to your local directory accordingly.
- Download the pre-trained feature backbone and transformer weights and set `PRETRAIN_BACKBONE_DIR` , `PRETRAIN_TRANSFORMER_DIR`(only for AVA dataset) accordingly. 
- make sure `LOAD` and `LOAD_FC` are set to False
  
Then run:
```
# run training from scratch
python3  train_tuber_jhmdb.py 
python3  train_tuber_ava.py 

```

### Test(prototype)
To Test, first modify the config file(in configuration,one for AVA, another one for JHMDB):
- set the correct `WORLD_SIZE`, `GPU_WORLD_SIZE`, `DIST_URL`, `WOLRD_URLS` based on experiment setup.
- set the `ANNO_PATH`, `DATA_PATH`(put only one avi or mp4 video,please) and `TEST_PATH`(path of the input video) to your local directory accordingly.
- Download the pre-trained model and set `PRETRAINED_PATH` to model path.
- make sure `LOAD` and `LOAD_FC` are set to True

Then set path to tuber and run:
```
# run evaluation
python3  detection_system.py

```



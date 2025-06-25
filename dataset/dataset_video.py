import cv2
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch
import imageio
from PIL import Image
import pickle
from .transforms import AlbumentationsAug, random_horizontal_flip, VideoRandomResizedCrop  # YOLOXHSVRandomAug
import pandas as pd
import random
import copy

from dataset.utils_dataset import (
    parse_annotation,
    pre_process_IMU_when_getitem,
    pre_process_annotation3D,
    pre_process_camera_pose,
    read_frame_from_mp4,
)

def get_win_frame_ids_str_from_annotation_frame_id_str(annotation_frame_id_str, config):
    frame_stride = config["DATASET"]["FRAME_STRIDE"]
    frame_num_per_window = config["DATASET"]["WINDOW_LENGTH"] * frame_stride

    frame_end = int(annotation_frame_id_str)
    frame_start = frame_end - (config["DATASET"]["WINDOW_LENGTH"] - 1) * frame_stride
    # like: [1,x,x],[1,x,x],...,[1,x,x],1
    win_frame_ids_str = [
        str(i) for i in range(frame_start, frame_end + 1, frame_stride)
    ]
    return win_frame_ids_str


class DemonProfVideoDataset(Dataset):
    def __init__(self, config, split="train", logger=None):
        self.config = config
        self.logger = logger
        self.split = split

        self.training_mode = True if split == "train" or split == "trainval" else False
        # self.training_mode = True if split == "train" or split == "val" or split == "trainval" else False

        self.logger.info(f"Building Demonstrator Proficiency Video Dataset {split} ...")

        # =============== adapt to train, val and test dataset ===============
        self.annotation_stride = None
        partial_take_num = None
        self.root_dir = config.DATASET.ROOT_DIR
        self.window_length = config.DATASET.WINDOW_LENGTH
        self.video_clip_len = config.DATASET.VIDEO_CLIP_LEN
        self.frame_stride = config.DATASET.FRAME_STRIDE

        if self.split in ["train", "trainval"]:
            # self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_TRAIN"]
            partial_take_num = config["DATASET"]["TAKE_NUM_TRAIN"]
        elif self.split == "val":
            # self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_VAL"]
            partial_take_num = config["DATASET"]["TAKE_NUM_VAL"]
        elif self.split == "test":
            # self.annotation_stride = 1
            partial_take_num = config["DATASET"]["TAKE_NUM_TEST"]

        #  =================== load annotation ==============
        self.annotation_path = (
            f"{config['DATASET']['ROOT_DIR']}/annotations/proficiency_demonstrator_{split}_v2.csv"
        )

        gt = pd.read_csv(self.annotation_path, header=None, sep='\s+', names=['take', 'label'])
        take_name_list = gt['take'].str.split('/').str[1].tolist()
        label = gt['label'].tolist()

        # if split == 'test':  # mask test set 
        #     label = list(np.array(label) * 0)

        self.annotations = dict(zip(take_name_list, label))

        # =============== check if takes folder exists ==============
        self.takes_dir = os.path.join(self.root_dir, config.DATASET.IMAGE_DIR) 
        self.valid_take_list = []
        invalid_count = 0

        for i, take_name in enumerate(take_name_list):
            take_dir = os.path.join(self.takes_dir, take_name)
            if os.path.exists(take_dir):
                self.valid_take_list.append(take_name)
            else:
                invalid_count += 1
                self.logger.info(f"take {take_name} does not exist, skipped")    

        print(f"valid takes count: {len(self.valid_take_list)}")

        # ================ image input ============
        # TODO: control whether to load images 

        # ================ pose input ==============
        self.use_pose_input = config.DATASET.get("USE_POSE_INPUT", False)
        self.pose_dir = os.path.join(self.root_dir, config.DATASET.get("POSE_DIR", None)) if self.use_pose_input else None  

        #  =============== valid_take_list ===============

        self.dummy_submission_path = os.path.join(self.root_dir, config.DATASET.DUMMY_SUMISSION_PATh)

        #   partial data
        if partial_take_num is not None:
            self.valid_take_list = self.valid_take_list[:partial_take_num]
            self.logger.info(f"Partial {split} data: {partial_take_num} takes")

        #  =============== Initial load data ===============
        self.logger.info(f"Start loading {split} data")

        if split in ["val", "test"] : 
            anno_path = config[split.upper()]["ANNOTATION_PATH"]

            with open(anno_path, 'rb') as file:
                self.sample_list = pickle.load(file)
            self.logger.info(f"Loaded {len(self.sample_list)} samples")

            sample_rate = config.DATASET.SAMPLE_RATE
            self.sample_list = self.sample_list[::sample_rate]

        else:
            # for debug
            ret = self._process_one_take(
                self.valid_take_list[0]
            )  # must run once to update self.config["DATASET"]["INPUT_DIM"]

            # Multi-processing
            with ThreadPoolExecutor(max_workers=config["WORKERS_PARALLEL"]) as executor:
                results = list(
                    tqdm(
                        executor.map(self._process_one_take, self.valid_take_list),
                        total=len(self.valid_take_list),
                        ncols=80,
                        position=0,
                        desc="Loading data",
                    )
                )

            # Merge results
            self.sample_list = []
            for result in results:
                self.sample_list.extend(result)

            repeated = config.DATASET.get("TRAIN_REPEATED", 1)
            self.sample_list = self.sample_list * repeated

        # Transform to tensor
        image_size = config.DATASET.get("IMAGE_SIZE", [224, 224])

        if self.training_mode:
            self.albumentation_aug = AlbumentationsAug()
            # self.yoloxhsvrandom_aug = YOLOXHSVRandomAug()
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.albumentation_aug = None
            # self.yoloxhsvrandom_aug = None
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.use_hflip = config.TRAIN.get("HORIZON_FLIP", False)
        self.use_randresizecrop = config.TRAIN.get("RANDOM_RESIZE_CROP", False) 
        self.rand_resize_crop = VideoRandomResizedCrop(
                                    size=224,
                                    scale=(0.2, 1.0),      # 裁剪区域为原图的20%-100%
                                    ratio=(3./4., 4./3.),  # 宽高比范围
                                    interpolation_mode='bilinear'
                                )

        self.logger.info(f'dataset build, total {len(self.sample_list)} samples')

    def _process_one_take(self, take_uid):

        sample_list = []

        # get a sample from the video
        ego_dir = os.path.join(self.takes_dir, take_uid, "ego")
        label = self.annotations[take_uid] 
        
        frames_list = os.listdir(ego_dir)
        num_frames = len(frames_list)
        frame_id = (num_frames - 1) * self.frame_stride

        # get images_path
        first_id = frame_id - (self.window_length - 1) * self.frame_stride
        last_id = frame_id + 1 * self.frame_stride
        win_frame_ids = list(range(first_id, last_id, self.frame_stride))
        images_path = [
            os.path.join(ego_dir, f"{frame_id:05d}.jpg")
            for frame_id in win_frame_ids
        ]
        exist_images_path = [
            os.path.exists(image_path) for image_path in images_path
        ]
        if not all(exist_images_path):
            print(f"Some images do not exist in {ego_dir}. skip this take. take_uid: {take_uid},  {frame_id}")
            # invalid_sample.append(images_path)
            for i, img in enumerate(frames_list):
                if not img.endswith('.jpg'):
                    print(i, img)
            return []

        sample = {
            "take_uid": take_uid,
            "frame_id": frame_id,
            "image_dir": ego_dir,
            "images_path": images_path,     
            "label": label,
        }

        sample_list.append(sample)  
        
        return sample_list

    def __len__(self):
        # return self.cumsum_window_num_per_data_list[-1]
        return len(self.sample_list)

    def __getitem__(self, index):

        ################ get sample  ################

        sample = copy.deepcopy(self.sample_list[index])

        if self.split in ["train", "trainval"]:   # random sampling a clip from one video for training
            take_uid = sample["take_uid"]
            image_dir = sample["image_dir"]
            
            frames_list = os.listdir(image_dir)
            num_frames = len(frames_list)

            # get images_path and frame_id
            frame_id = (random.randint(self.window_length, num_frames) - 1) * self.frame_stride

            first_id = frame_id - (self.window_length - 1) * self.frame_stride
            last_id = frame_id + 1 * self.frame_stride
            win_frame_ids = list(range(first_id, last_id, self.frame_stride))
            images_path = [
                os.path.join(image_dir, f"{frame_id:05d}.jpg")
                for frame_id in win_frame_ids
            ]

            sample["images_path"] = images_path
            sample["frame_id"] = frame_id
        
        # get win_frame_ids
        frame_id = sample["frame_id"]
        first_id = frame_id - (self.window_length - 1) * self.frame_stride
        last_id = frame_id + 1 * self.frame_stride
        win_frame_ids = list(range(first_id, last_id, self.frame_stride))

        # get the inputs_image
        inputs_image = []           
        images_path = sample['images_path'][-self.video_clip_len:]
        
        for img_path in images_path:

            if not os.path.exists(img_path):
                print(f"{img_path} does not exist. skip this take. take_uid: {data_dir['take_uid']}")
                # return None
            img =  Image.open(img_path)
        
        # 训练时 数据增强
            # if self.training_mode:
            #     # 应用 Albumentations 数据增强
            #     img = np.array(img)
            #     # img = self.yoloxhsvrandom_aug(img)  # no use
            #     img = self.albumentation_aug(img)

            #     # 将增强后的 numpy 数组转换回 PIL 图片，以便使用 torchvision.transforms
            #     img = Image.fromarray(img)

            img_tensor = self.transform(img)
            inputs_image.append(img_tensor)            

        input_image_tensors = torch.stack(inputs_image, dim=0)   # (16, 3, 224, 224)

        if self.training_mode:
            if self.use_hflip:
                input_image_tensors = random_horizontal_flip(input_image_tensors)
            if self.use_randresizecrop:
                input_image_tensors = self.rand_resize_crop(input_image_tensors)

        ################ get the inputs_pose ################
        if self.use_pose_input: 
            inputs_pose = []
            pose_paths = [
                os.path.join(self.pose_dir, sample['take_uid'], 'ego', f"{frame_id:05d}.npy")
                for frame_id in win_frame_ids
            ]   
            pose_paths = pose_paths[-self.video_clip_len:]

            for pose_path in pose_paths:
                
                if not os.path.exists(img_path):
                    print(f"{img_path} does not exist. skip this take. take_uid: {data_dir['take_uid']}")
                    pose = np.zeros((51, ))
                else:
                    pose = np.load(pose_path)
                    pose = pose.reshape(51)
                inputs_pose.append(pose)
            
            inputs_pose = np.stack(inputs_pose, axis=0).astype(np.float32)
        else:
            inputs_pose = np.zeros((self.video_clip_len, 51))

        ################ get the label ################
        label = sample['label']

        return {
            "take_uid": sample["take_uid"], 
            "frame_id": sample["frame_id"],
            "inputs_image": input_image_tensors.numpy(), 
            "inputs_pose": inputs_pose, 
            "label": label, # np.array(target, dtype=np.float32),
        }

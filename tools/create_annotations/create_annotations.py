import os
import sys
sys.path.append(os.getcwd())
paths = sys.path
for path in paths:
    print(path)

import cv2
from torch.utils.data import Dataset

import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd

from utils.utils import get_config, get_logger_and_tb_writer
import pickle

class DemonstratorProficiencyDataset(Dataset):
    def __init__(self, config, split="train", logger=None):
        self.config = config
        self.logger = logger
        self.split = split

        self.training_mode = True if split == "train" or split == "val" or split == "trainval" else False

        self.logger.info(f"Building DemonstratorProficiencyDataset {split} ...")

        # =============== adapt to train, val and test dataset ===============
        self.annotation_stride = None
        partial_take_num = None
        self.root_dir = config.DATASET.ROOT_DIR
        self.window_length = config.DATASET.WINDOW_LENGTH
        self.video_clip_len = config.DATASET.VIDEO_CLIP_LEN
        self.frame_stride = config.DATASET.FRAME_STRIDE

        if self.split == "train":
            self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_TRAIN"]
            partial_take_num = config["DATASET"]["TAKE_NUM_TRAIN"]
        elif self.split == "val":
            self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_VAL"]
            partial_take_num = config["DATASET"]["TAKE_NUM_VAL"]
        elif self.split == "trainval":
            self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_TRAINVAL"]
        elif self.split == "test":
            self.annotation_stride = 1
            partial_take_num = config["DATASET"]["TAKE_NUM_TEST"]

        #  =================== load annotation ==============
        self.annotation_path = (
            f"{config['DATASET']['ROOT_DIR']}/annotations/proficiency_demonstrator_{split}_v2.csv"
        )

        gt = pd.read_csv(self.annotation_path, header=None, sep='\s+', names=['take', 'label'])
        take_name_list = gt['take'].str.split('/').str[1].tolist()
        label = gt['label'].tolist()

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

        #  =============== take_uid_list ===============

        # take_uid_list = []
        self.dummy_submission_path = os.path.join(self.root_dir, config.DATASET.DUMMY_SUMISSION_PATh)

        #   partial data
        if partial_take_num is not None:
            self.valid_take_list = self.valid_take_list[:partial_take_num]
            self.logger.info(f"Partial {split} data: {partial_take_num} takes")

        #  =============== Initial load data ===============
        self.logger.info(f"Start loading {split} data")

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


        sample_rate = config.DATASET.SAMPLE_RATE

        # save annotations to file
        save_path = os.path.join('./',  f"annotations_{split}.pkl")

        self.logger.info(f"Saving annotations to {save_path}")  
        with open(save_path, "wb") as f:
            pickle.dump(self.sample_list, f)

        # self.sample_list = self.sample_list[::sample_rate]

        self.logger.info(f'dataset build, total {len(self.sample_list)} samples')
    def _process_one_take(self, take_uid):

        ego_dir = os.path.join(self.takes_dir, take_uid, "ego")
        frames_list = os.listdir(ego_dir)
        num_frames = len(frames_list)

        sample_list = []
        invalid_sample = []

        for i in range(self.window_length-1, num_frames):
            frame_id = i * self.frame_stride
            label = self.annotations[take_uid]

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
                # print(f"Some images do not exist in {ego_dir}. skip this take. take_uid: {take_uid},  {frame_id}")
                invalid_sample.append(images_path)
                continue

            sample = {
                "take_uid": take_uid,
                "frame_id": frame_id,
                "image_dir": ego_dir,
                "images_path": images_path,     
                "label": label,
            }

            sample_list.append(sample)

        self.logger.info(f"take '{take_uid}' has {len(sample_list)} samples")      

        return sample_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="tools/create_annotations/create_annotations.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    args = parser.parse_args()
    
    config = get_config(args)
    logger, tb_writer = get_logger_and_tb_writer(config)
    
    dataset_train = DemonstratorProficiencyDataset(config, split="train", logger=logger)
    dataset_val = DemonstratorProficiencyDataset(config, split="val", logger=logger)
    dataset_test = DemonstratorProficiencyDataset(config, split="test", logger=logger)
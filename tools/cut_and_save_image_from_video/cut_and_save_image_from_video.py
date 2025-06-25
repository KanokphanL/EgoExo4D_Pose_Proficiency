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

# from dataset.utils_dataset import (
#     parse_annotation,
#     pre_process_IMU_when_getitem,
#     pre_process_annotation3D,
#     pre_process_camera_pose,
#     read_frame_from_mp4,
# )
from utils.utils import get_config, get_logger_and_tb_writer

# def get_win_frame_ids_str_from_annotation_frame_id_str(annotation_frame_id_str, config):
#     frame_stride = config["DATASET"]["FRAME_STRIDE"]
#     frame_num_per_window = config["DATASET"]["WINDOW_LENGTH"] * frame_stride

#     frame_end = int(annotation_frame_id_str)
#     frame_start = frame_end - (config["DATASET"]["WINDOW_LENGTH"]-1) * frame_stride
#     # like: [1,x,x],[1,x,x],...,[1,x,x],1
#     win_frame_ids_str = [str(i) for i in range(frame_start, frame_end + 1, frame_stride)]
#     return win_frame_ids_str
    
def video_to_jpg(video_path, img_path, frame_stride=3):
    # 创建保存图片的目录
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_stride == 0:
            # 使用 frame_count 命名文件并格式化
            img_name = os.path.join(img_path, f"{str(frame_count).zfill(5)}.jpg")
            cv2.imwrite(img_name, frame)

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    # print(f"视频 {video_path} 处理完成，共保存{frame_count/frame_stride}图片到 {img_path}")


class BodyPoseDataset(Dataset):
    def __init__(self, config, split="train", logger=None):
        self.config = config
        self.logger = logger
        self.split = split

        # =============== adapt to train, val and test dataset ===============
        # self.annotation_stride = None
        partial_take_num = config.DATASET.TAKE_NUM_TRAIN

        self.data_path = config.DATASET.ROOT_DIR
        self.takes_path = os.path.join(self.data_path, 'takes')
        self.frame_stride = config.DATASET.FRAME_STRIDE

        #  =============== camera_pose_dir ===============
        # self.camera_pose_dir = None

        # self.camera_pose_dir = (
        #     f"{config['DATASET']['ROOT_DIR']}/annotations/ego_pose/{split}/camera_pose"
        # )

        #  =============== take_uid_list ===============
        take_uid_list = []
        self.test_dir = {}
            
        takes_paths = []
        takes_paths = os.listdir(self.takes_path)

        take_uid_list = [file_path for file_path in takes_paths]
            
        #   partial data
        if partial_take_num is not None:
            take_uid_list = take_uid_list[:partial_take_num]
            self.logger.info(f"Partial {split} data: {partial_take_num} takes")

        #  =============== Initial load data ===============
        self.logger.info(f"Start loading {split} data")

        # for debug
        self._process_take_uid(
            take_uid_list[0]
        )  # must run once to update self.config["DATASET"]["INPUT_DIM"]

        # Multi-processing
        with ThreadPoolExecutor(max_workers=config["WORKERS_PARALLEL"]) as executor:
            results = list(
                tqdm(
                    executor.map(self._process_take_uid, take_uid_list),
                    total=len(take_uid_list),
                    ncols=80,
                    position=0,
                    desc="Loading data",
                )
            )

        # Merge results
        # self.data_dir_list = []
        # self.window_num_per_data_list = []
        # self.cumsum_window_num_per_data_list = []

        print(f"Done {len(take_uid_list)} takes processed.")

    def _process_take_uid(self, take_uid):
        # # ===== get annotation_frame_ids_str, in which each element corresponds to a sample
        # annotation_frame_ids_str = []
                            
        # # ===== read and save images from mp4
        # inputs_image_in_all_win = []
                    
        take_path = ego_path = os.path.join(self.takes_path, take_uid, 'frame_aligned_videos/downscaled/448/')
        ego_path = os.path.join(take_path, 'aria01_214-1.mp4')

        if os.path.exists(ego_path) is False:

            files = os.listdir(take_path)

            valid_files = []
            for file in files:
            # 检查文件名是否符合 ari*_214-1.mp4 模式
                if file.startswith('ari') and file.endswith('_214-1.mp4'):
                    # 拼接完整的文件路径
                    print(f"{take_path}, found {file}")
                    valid_files.append(os.path.join(take_path, file))

            if len(valid_files) > 0:
                ego_path = valid_files[0]
            else:
                tqdm.write(f"video file not found: {ego_path}, skip this take")
                return None

        exo1_path = os.path.join(take_path, 'cam01.mp4')

        if os.path.exists(exo1_path) is False:
            exo1_path = os.path.join(take_path, 'gp01.mp4')
            exo2_path = os.path.join(take_path, 'gp02.mp4')
            exo3_path = os.path.join(take_path, 'gp03.mp4')
            exo4_path = os.path.join(take_path, 'gp04.mp4')
        else:
            exo1_path = os.path.join(take_path, 'cam01.mp4')
            exo2_path = os.path.join(take_path, 'cam02.mp4')
            exo3_path = os.path.join(take_path, 'cam03.mp4')
            exo4_path = os.path.join(take_path, 'cam04.mp4')
        
        # read images from mp4

        
        ego_img_path = f"{self.data_path}/takes_image_downscaled_448/{take_uid}/ego"
        exo1_img_path = f"{self.data_path}/takes_image_downscaled_448/{take_uid}/exo1"
        exo2_img_path = f"{self.data_path}/takes_image_downscaled_448/{take_uid}/exo2"
        exo3_img_path = f"{self.data_path}/takes_image_downscaled_448/{take_uid}/exo3"
        exo4_img_path = f"{self.data_path}/takes_image_downscaled_448/{take_uid}/exo4"
        
        # ego image
        image_paths = [ego_img_path, ]    
        videos = [ego_path,]
        # image_paths = [ego_img_path, exo1_img_path, exo2_img_path, exo3_img_path, exo4_img_path]    
        # videos = [ego_path, exo1_path, exo2_path, exo3_path, exo4_path]

        for video_path, img_path in zip(videos, image_paths):
            # print(video_path, img_path)
            video_to_jpg(video_path, img_path)


 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="tools/cut_and_save_image_from_video/00_cut_and_save_image_from_video.yaml",
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
    
    dataset_train = BodyPoseDataset(config, split="train", logger=logger)
    # dataset_val = BodyPoseDataset(config, split="val", logger=logger)
    # dataset_test = BodyPoseDataset(config, split="test", logger=logger)
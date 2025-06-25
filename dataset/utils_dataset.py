from concurrent.futures import ProcessPoolExecutor
import functools
import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
import copy

def parse_annotation(annotation_dir, mode="3D"):
    joint_to_index = {
        "nose": 0,
        "left-eye": 1,
        "right-eye": 2,
        "left-ear": 3,
        "right-ear": 4,
        "left-shoulder": 5,
        "right-shoulder": 6,
        "left-elbow": 7,
        "right-elbow": 8,
        "left-wrist": 9,
        "right-wrist": 10,
        "left-hip": 11,
        "right-hip": 12,
        "left-knee": 13,
        "right-knee": 14,
        "left-ankle": 15,
        "right-ankle": 16,
    }
    if mode == "3D":
        annotation = np.zeros((17, 3))
    elif mode == "2D":
        annotation = np.zeros((17, 2))

    mask = np.zeros((17, 1))
    for joint in annotation_dir:
        mask[joint_to_index[joint]] = 1
        annotation[joint_to_index[joint]][0] = annotation_dir[joint]["x"]
        annotation[joint_to_index[joint]][1] = annotation_dir[joint]["y"]
        if mode == "3D":
            annotation[joint_to_index[joint]][2] = annotation_dir[joint]["z"]

    return annotation, mask


def read_frame_from_mp4(input, frame_ids, release=True):
    if type(input) == str:
        mp4_filepath = input
        cap = cv2.VideoCapture(mp4_filepath)
        if not cap.isOpened():
            tqdm.write(f"Could not open video.{mp4_filepath}")
            exit()
    elif type(input) == cv2.VideoCapture:
        cap = input
        
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 确保要提取的帧索引在视频的总帧数范围内
    frame_ids_valid = [i if i > 0 else 0 for i in frame_ids]
    frame_ids_valid = [i if i < total_frames else total_frames - 1 for i in frame_ids]
    
    # 读取并保存指定帧
    frames = []
    for frame_index in frame_ids_valid:
        # 设置视频的当前帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取当前帧
        ret, frame = cap.read()

        if ret:
            frames.append(frame)
            # 显示当前帧（可选）
            # cv2.imshow(f'Frame {frame_index}', frame)
            # cv2.waitKey(0)  # 按任意键关闭当前帧显示窗口

            # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            # cv2.startWindowThread()
            # cv2.imshow('Image', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
        else:
            tqdm.write(f"read_frame_from_mp4: Could not read frame {frame_index}.")
    if release:
        cap.release()

    return frames


"""
Pre-process the input data.

input_dir["K"].shape
torch.Size([16, 3, 3])
input_dir["R"].shape
torch.Size([16, 20, 3, 3])
input_dir["T_transposed"].shape
torch.Size([16, 20, 1, 3])
target_dir["annotation3D"].shape
torch.Size([16, 20, 17, 3])
target_dir["mask3D"].shape
torch.Size([16, 20, 17, 1])
target_dir["annotation2D"].shape
torch.Size([16, 20, 17, 2])
target_dir["mask2D"].shape
torch.Size([16, 20, 17, 1])
# 16: batch size, 20: sequence length, 17: number of joints


"""


def pre_process_camera_pose(camera_extrinsics, config):
    """
    camera_extrinsics.shape
    [N_frame, 3, 4]
    type(camera_extrinsics) == np.ndarray
    """
    T_transposed = camera_extrinsics[:, :, -1] # [N_frame, 3]
    R = camera_extrinsics[:, :, :-1] # [N_frame, 3, 3]
    R_transposed = np.transpose(R, (0, 2, 1))

    mode = config["PRE_PROCESS"]["INPUT_IMU_MODE"]
    if mode == "T":
        input_IMU = T_transposed
        
    elif mode == "R_and_T":
        input_IMU = np.concatenate((T_transposed, R_transposed.reshape(-1, 9)), axis=-1) # [N_frame, 12]
        
    elif mode == "imu_point":
        # input_IMU = np.einsum(
        #     "ijk,ikl->ijl", -T_transposed, np.linalg.inv(R_transposed)
        # )
        input_IMU = -T_transposed @ np.linalg.inv(R_transposed) # [N_frame, 3]

    elif mode == "imu_vector":
        input_IMU_0 = -T_transposed @ np.linalg.inv(R_transposed)
        input_IMU_z = (np.array([0, 0, 1]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU = np.concatenate((input_IMU_0, input_IMU_z), axis=-1) # [N_frame, 6]
        
    elif mode == "imu_vector_xyz":
        input_IMU_0 = -T_transposed @ np.linalg.inv(R_transposed)
        input_IMU_x = (np.array([1, 0, 0]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU_y = (np.array([0, 1, 0]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU_z = (np.array([0, 0, 1]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU = np.concatenate((input_IMU_0, input_IMU_x, input_IMU_y, input_IMU_z), axis=-1) # [N_frame, 12]
        
    ##################### Augumentation using diff #####################
    diff_interval = config["PRE_PROCESS"]["INPUT_IMU_DIFF_INTERVAL"]
    if diff_interval is not None and diff_interval > 0:
        input_IMU_diff = np.concatenate((np.zeros((diff_interval, input_IMU.shape[-1])), input_IMU[diff_interval:] - input_IMU[:-diff_interval]), axis=0)
        input_IMU = np.concatenate((input_IMU, input_IMU_diff), axis=-1) # [N_frame, x] ->  [N_frame, 2x]
        
    
    ##################### change INPUT_DIM in config #####################
    config["DATASET"]["INPUT_DIM"] = input_IMU.shape[-1]
    return input_IMU


def pre_process_annotation3D(camera_extrinsics, annotation3D_world, config):
    """
    camera_extrinsics.shape
    [3, 4]
    type(camera_extrinsics) == np.ndarray
    annotation3D_world.shape
    [17, 3]
    type(annotation3D_world) == np.ndarray
    """
    T_transposed = camera_extrinsics[None, :, -1]
    R_transposed = camera_extrinsics[:, :-1].T

    target_mode = config["PRE_PROCESS"]["TARGET_MODE"]
    if target_mode == "camera":
        annotation3D = annotation3D_world @ R_transposed + T_transposed
    elif target_mode == "world":
        annotation3D = annotation3D_world

    return annotation3D

def pre_process_IMU_when_getitem(inputs, config):
    """
    inputs: [N_frame, x] or [N_frame, 2x], x = 3, 6, 12
    """
    mode = config["PRE_PROCESS"]["INPUT_IMU_MODE"]
    
    mean_xyz = np.mean(inputs[:, :3], axis=0)
    mean_xyz = np.tile(mean_xyz, inputs.shape[-1] // 3)
    
    diff_interval = config["PRE_PROCESS"]["INPUT_IMU_DIFF_INTERVAL"]
    if diff_interval is not None and diff_interval > 0:
        mean_xyz[inputs.shape[-1] // 2:] = 0
    
    return inputs - mean_xyz
        
def _process_one_take(key_value_pair, config):        
    # for take_uid in tqdm(result_json, ncols=80, position=0, desc="Post-processing"):
    take_uid, value = key_value_pair
    data_root = config["DATASET"]["ROOT_DIR"]
    target_mode = config["PRE_PROCESS"]["TARGET_MODE"]
    
    splits = ["train", "val", "test"]
    camera_path = None
    for split in splits:
        path = f"{data_root}/annotations/ego_pose/{split}/camera_pose/{take_uid}.json"
        if os.path.exists(path):
            camera_path = path
            break
    if camera_path is None:
        tqdm.write(f"camera path not found for {take_uid}")
        return take_uid, None

    camera_json = json.load(open(os.path.join(camera_path)))
    for key in camera_json.keys():
        if "aria" in key:
            aria_key = key
            break
    for frame_id_str in value["body"]:
        camera_extrinsics = np.array(
            camera_json[aria_key]["camera_extrinsics"][frame_id_str]
        )
        T_transposed = camera_extrinsics[None, :, -1]
        R_transposed = camera_extrinsics[:, :-1].T
        body_joints = np.array(value["body"][frame_id_str])
        if target_mode == "camera":
            value["body"][frame_id_str] = (
                (body_joints - T_transposed) @ np.linalg.inv(R_transposed)
            ).tolist()

    return take_uid, value

def post_process_result_by_participant(input_dict, config):

    ### postprocess - aggregrate the results by participant uid ###

    # get take_name to participant id

    takes_path = os.path.join(config.DATASET.ROOT_DIR, "takes.json")
    with open(takes_path, "r") as fp:
        takes = json.load(fp)

    # 提取结果字典中的信息
    video_names = input_dict["videos"]
    original_scores = input_dict["scores"]
    original_predictions = input_dict["predictions"]

    # 创建视频名称到索引的映射
    video_to_idx = {name: i for i, name in enumerate(video_names)}
    
    # 过滤出现在结果字典中的视频的takes信息
    valid_takes = [take for take in takes if take["take_name"] in video_to_idx]

    # 按参与者ID收集视频索引
    participant_to_video_indices = defaultdict(list)
    for take in valid_takes:
        video_name = take["take_name"]
        participant_uid = take["participant_uid"]
        video_idx = video_to_idx[video_name]
        participant_to_video_indices[participant_uid].append(video_idx)

    # 计算每个参与者的平均分数和预测类别
    participant_scores = {}
    participant_predictions = {}
    
    for participant, video_indices in participant_to_video_indices.items():
        # 收集该参与者所有视频的分数
        scores = [original_scores[idx] for idx in video_indices]
        if not scores:
            continue
            
        # 计算平均分数
        num_classes = len(scores[0])
        avg_score = [sum(score[i] for score in scores) / len(scores) for i in range(num_classes)]
        participant_scores[participant] = avg_score
        
        # 计算基于平均分数的预测类别
        prediction = avg_score.index(max(avg_score))
        participant_predictions[participant] = prediction

    # 更新结果字典中的分数和预测类别
    updated_scores = copy.deepcopy(original_scores)
    updated_predictions = copy.deepcopy(original_predictions)
    
    # 对每个参与者，更新其所有视频的分数和预测类别
    for participant, video_indices in participant_to_video_indices.items():
        if participant in participant_scores:
            # 获取该参与者的聚合分数和预测类别
            score = participant_scores[participant]
            prediction = participant_predictions[participant]
            
            # 更新该参与者所有视频的分数和预测类别
            for idx in video_indices:
                updated_scores[idx] = score
                updated_predictions[idx] = prediction

    # 更新结果字典
    result_dict = copy.deepcopy(input_dict)
    result_dict["scores"] = updated_scores
    result_dict["predictions"] = updated_predictions
        
    return result_dict


def post_process_participant_by_voting(result_dict, config):
    ### postprocess - aggregate the results by participant uid ###

    # get take_name to participant id
    takes_path = os.path.join(config.DATASET.ROOT_DIR, "takes.json")
    with open(takes_path, "r") as fp:
        takes = json.load(fp)

    # 提取结果字典中的信息
    video_names = result_dict["videos"]
    original_predictions = result_dict["predictions"]

    # 创建视频名称到索引的映射
    video_to_idx = {name: i for i, name in enumerate(video_names)}
    
    # 过滤出现在结果字典中的视频的takes信息
    valid_takes = [take for take in takes if take["take_name"] in video_to_idx]

    # 按参与者ID收集视频索引
    participant_to_video_indices = defaultdict(list)
    for take in valid_takes:
        video_name = take["take_name"]
        participant_uid = take["participant_uid"]
        video_idx = video_to_idx[video_name]
        participant_to_video_indices[participant_uid].append(video_idx)

    # 计算每个参与者的投票结果
    participant_predictions = {}
    
    for participant, video_indices in participant_to_video_indices.items():
        # 收集该参与者所有视频的预测类别
        predictions = [original_predictions[idx] for idx in video_indices]
        if not predictions:
            continue
            
        # 使用投票机制确定最终预测类别
        vote_counts = Counter(predictions)
        max_votes = max(vote_counts.values())
        top_classes = [cls for cls, cnt in vote_counts.items() if cnt == max_votes]
        final_class = min(top_classes)  # 若有平局，选择编号最小的类别
        
        participant_predictions[participant] = final_class

    # 更新结果字典中的预测类别
    updated_predictions = copy.deepcopy(original_predictions)
    
    # 对每个参与者，更新其所有视频的预测类别
    for participant, video_indices in participant_to_video_indices.items():
        if participant in participant_predictions:
            # 获取该参与者的投票结果
            prediction = participant_predictions[participant]
            
            # 更新该参与者所有视频的预测类别
            for idx in video_indices:
                updated_predictions[idx] = prediction

    # 更新结果字典
    ret = copy.deepcopy(result_dict)
    ret['videos'] = video_names
    ret['predictions'] = updated_predictions

    return ret
    # result_dict["predictions"] = updated_predictions
    # return result_dict
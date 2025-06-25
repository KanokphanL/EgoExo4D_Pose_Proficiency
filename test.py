import argparse
import json
import os
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from dataset.utils_dataset import post_process_result_by_participant, post_process_participant_by_voting
from utils.utils import (
    get_config,
    get_logger_and_tb_writer,
    load_model,
)
from dataset.dataset import DemonstratorProficiencyDataset
from dataset.dataset_video import DemonProfVideoDataset

# from models.model_Baseline import Baseline
from models.ego_video_resnet import EgoVideoResnet3D
from models.ego_fusion import EgoFusion

def get_test_loader(config, logger, is_distributed=False):
    dataset_name = config.DATASET.get("NAME", "DemonstratorProficiencyDataset")
    dataset = eval(dataset_name)(config, split="test", logger=logger)
    sampler = torch.utils.data.DistributedSampler(dataset) if is_distributed else None
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["TEST"]["BATCH_SIZE"],
        shuffle=False,
        sampler=sampler,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=False,
        pin_memory=True,
    )
    return test_loader

def aggregate_video_scores(videos: List[str], scores: List[List[float]]) -> Tuple[List[str], List[List[float]], List[int]]:
    """
    按视频名称对clip预测分数进行分组平均，得到每个视频的最终分数，并计算预测类别
    
    参数:
    videos (List[str]): 每个clip对应的视频名称列表
    scores (List[List[float]]): 每个clip的预测分数列表，每个分数是一个长度为num_classes的列表
    
    返回:
    Tuple[List[str], List[List[float]], List[int]]: 处理后的视频名称列表、对应的平均分数列表和预测类别列表
    """
    # 检查输入是否有效
    if len(videos) != len(scores):
        raise ValueError("视频名称列表和分数列表长度必须一致")
    
    # 用于存储每个视频的所有clip分数
    video_clips: Dict[str, List[List[float]]] = defaultdict(list)
    
    # 按视频名称收集所有clip的分数
    for video_name, score in zip(videos, scores):
        video_clips[video_name].append(score)
    
    # 保持原始视频顺序的列表
    unique_videos = []
    seen = set()
    for video in videos:
        if video not in seen:
            seen.add(video)
            unique_videos.append(video)
    
    # 计算每个视频的平均分数和预测类别
    new_scores = []
    new_predictions = []
    for video in unique_videos:
        clips = video_clips[video]
        # 对每个类别分别求平均
        avg_score = [sum(clip[i] for clip in clips) / len(clips) for i in range(len(clips[0]))]
        new_scores.append(avg_score)
        # 计算预测类别（最大分数对应的类别索引）
        prediction = avg_score.index(max(avg_score))
        new_predictions.append(prediction)
    
    return unique_videos, new_scores, new_predictions

# def calculate_accuracy(video_names: List[str], predictions: List[int], groundtruth: Dict[str, int]) -> float:
def calculate_accuracy(video_names, predictions, groundtruth):

    """
    计算分类准确率
    
    参数:
    video_names (List[str]): 视频名称列表
    predictions (List[int]): 预测类别列表
    groundtruth (Dict[str, int]): 真实标签字典，键为视频名称，值为真实类别
    
    返回:
    float: 分类准确率
    """
    if len(video_names) != len(predictions):
        raise ValueError("视频名称列表和预测结果列表长度必须一致")
    
    correct = 0
    total = 0
    
    for video_name, pred in zip(video_names, predictions):
        if video_name in groundtruth:
            total += 1
            if pred == groundtruth[video_name]:
                correct += 1
    
    return correct / total if total > 0 else 0.0

def test(
    config,
    device,
    test_loader,
    model
):
    model.eval()

    labels = []
    preds = []
    
    videos = []
    predictions = []
    scores = []

    # 初始化 tqdm 进度条
    progress_bar = tqdm(test_loader, ncols=80, position=0)
    progress_bar.set_description("Testing")
    with torch.no_grad():
        for index, data_dir in enumerate(progress_bar):

            # parse data from dataloader
            inputs_image = data_dir["inputs_image"].to(device)
            inputs_pose = data_dir["inputs_pose"].to(device)    
            label = data_dir["label"].to(device)

            # forward
            pred = model(inputs_image, inputs_pose)

            preds.append(pred)
            labels.append(label)

            # save result
            batch_size = pred.size(0)
            score = torch.nn.functional.softmax(pred, dim=1)    
            for i in range(batch_size):
                take_uid = data_dir["take_uid"][i]
                videos.append(take_uid)
                
                prediction = torch.argmax(pred[i], dim=0).cpu().detach().numpy().item()
                predictions.append(prediction)

                scores.append(score[i].cpu().detach().numpy().tolist())        

    # get video predictions by uid
    unique_videos, video_scores, video_prediction = aggregate_video_scores(videos, scores)

    # compute the accuracy
    groundtruth = test_loader.dataset.annotations
    accuracy = calculate_accuracy(unique_videos, video_prediction, groundtruth)

    test_result_json = {
        "videos": unique_videos,        
        "predictions": video_prediction,
        "scores": video_scores,
    }

    return test_result_json, accuracy

def process_and_save_result(config, test_result_json, save_path):
    ################## post processing ##################
    # test_result_json = post_process_result(test_result_json, config)
    
    with open(save_path, "w") as f:
        json.dump(test_result_json, f)
    
def main(args, logger=None):

    config = get_config(args)
    if logger is None:
        logger, _ = get_logger_and_tb_writer(config, split="test")

    logger.info(f"config: {config}")
    logger.info(f"args: {args}")

    ################## dataloader ##################
    config["DATASET"]["TAKE_NUM_TEST"] = None # use all data
    test_loader = get_test_loader(config, logger)

    ################## device ##################
    # conside multi-gpu training and validation
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    ################## model ##################
    if config["SELECT_MODEL"] == "BASELINE":
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = Baseline(config).to(device)
    else:
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = eval(config.SELECT_MODEL)(config)
        model = model.to(device)

    model_path = args.model_path
    logger.info(f"Load pretrained model: {model_path}")
    try:
        model.load_state_dict(load_model(model_path), strict=False)
    except:
        logger.error(f"Could not load pretrained model: {model_path}")

    ################## test ##################
    result_dict, test_acc = test(
        config,
        device,
        test_loader,
        model    
    )

    logger.info(f"Test acc: {test_acc:.4f}")

    ################## save result ##################
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    test_result_json_name = f"test_{model_name}_test-acc-{test_acc:.3f}.json"
    save_path = os.path.join(model_dir, test_result_json_name)
    
    with open(save_path, "w") as f:
        json.dump(result_dict, f)
    logger.info(f"Save result: {save_path}")

    ################## post processing and save result ##################

    accuracy_list = []
    result_list = []

    # post processing by scores and compute the accuracy
    post_result_dict = post_process_result_by_participant(result_dict, config)
    
    groundtruth = test_loader.dataset.annotations
    accuracy_after = calculate_accuracy(
        post_result_dict['videos'], 
        post_result_dict['predictions'], 
        groundtruth)
    logger.info(f"Test set accuracy after postprocess by scores:  {accuracy_after:.4f}")
    accuracy_list.append(accuracy_after)
    result_list.append(post_result_dict)

    # post process by voting and compute the accuracy
    post_result_dict = post_process_participant_by_voting(result_dict, config)
    accuracy_after = calculate_accuracy(post_result_dict['videos'], post_result_dict['predictions'], groundtruth)
    logger.info(f"Test set accuracy after poseprocess by voting:  {accuracy_after:.4f}")

    accuracy_list.append(accuracy_after)
    result_list.append(post_result_dict)

    # save the result
    print("-----------------------save the result------------------------------------")
    print(accuracy_list)

    # 找到最大准确率对应的索引
    max_index = accuracy_list.index(max(accuracy_list))
    
    # 获取最佳结果
    best_result = result_list[max_index]
    best_accuracy = accuracy_list[max_index]

    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    test_result_json_name = f"test_postprocessing_{model_name}_test-acc-{best_accuracy:.3f}.json"
    save_path = os.path.join(model_dir, test_result_json_name)
    
    with open(save_path, "w") as f:
        json.dump(best_result, f)
    logger.info(f"Save result: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="official_model/v0_official_baseline.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--model_path",
        default="official_model/baseline.pth",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    args = parser.parse_args()
    main(args)

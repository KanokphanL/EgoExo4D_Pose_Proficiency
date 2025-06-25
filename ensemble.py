import json
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from utils.utils import (
    get_config,
    get_logger_and_tb_writer,
    load_model,
)
import argparse
# from models.loss import MPJPELoss
from dataset.dataset import DemonstratorProficiencyDataset
from test import calculate_accuracy

from dataset.utils_dataset import post_process_result_by_participant, post_process_participant_by_voting

### Validation Set
val_weights = [1, 1, 1, 1]

val_prediction_files = [
    # 'val_output/val_v1_best-e6_val-acc-0.51.json', 
    # 'val_output/val_v1-best-e1-train-0.21-val-2.97.json',
    'val_output/val_v10_best-e9_val-acc-0.46_val-acc-0.456.json', 
    'val_output/val_v13-best-e1-train-0.45-val-2.23_val-acc-0.398.json',
    'val_output/val_v15_best-e2_val-acc-0.48_val-acc-0.481.json',   

]

### Test set
test_weights = [1, 1, 1, 1, 1, 1, 1]
# test_weights = val_weights

test_prediction_files = [
    # 'test_output/test_v1-best-e1-train-0.21-val-2.97_test-0.474_test-acc-0.42.json', 
    # 'test_output/test_v3_best-e9_val-acc-0.44_test-acc-0.42.json', 
    # 'test_output/test_v10_best-e9_val-acc-0.46_test-acc-0.458.json',
    # 'test_output/test_v12-best-e1-train-0.47-val-2.26_test-acc-0.415.json',
    # 'test_output/test_v13-best-e1-train-0.45-val-2.23_test-acc-0.462.json',
    # 'test_output/test_v15_best-e2_val-acc-0.48_test-acc-0.486.json',
    'test_output/test_v15_best-e4_val-acc-0.46_test-acc-0.482.json',
    # 'test_output/test_v17_best-e13_val-acc-0.45_test-acc-0.410.json',
    # 'test_output/test_v19_best-e4_val-acc-0.48_test-acc-0.453.json',
    'test_output/test_v19-best-e1-train-1.14-val-1.33_test-acc-0.488.json',
    # 'test_output/test_v20-best-e2-train-0.91-val-1.35_test-acc-0.475.json',
    # 'test_output/test_v20_best-e6_val-acc-0.47_test-acc-0.449.json',
    'test_output/test_v24_best-e6_val-acc-0.47_test-acc-0.486.json',
    # 'test_output/test_v24-best-e1-train-1.22-val-1.21_test-acc-0.452.json',
    # 'test_output/test_v25_best-e4_val-acc-0.45_test-acc-0.447.json',
    'test_output/test_v26_best-e5_val-acc-0.47_test-acc-0.516.json',
    'test_output/test_v26_best-e5_val-acc-0.47_test-acc-0.516.json',
    'test_output/test_v27_best-e3_val-acc-0.47_test-acc-0.514.json',
    'test_output/test_v27_best-e3_val-acc-0.47_test-acc-0.514.json',
    # 'test_output/test_v16-best-e3-train-1.05-val-1.29_test-acc-0.391.json', #'test_output/test_v16_best-e16_val-acc-0.48_test-acc-0.401.json',
]

# test_prediction_files = [
#     'test_output/test_postprocessing_v10_best-e9_val-acc-0.46_test-acc-0.495.json', 
#     'test_output/test_postprocessing_v13-best-e1-train-0.45-val-2.23_test-acc-0.495.json', 
#     'test_output/test_postprocessing_v15_best-e2_val-acc-0.48_test-acc-0.503.json', 
# ]


def get_val_loader(config, split, logger):
    val_dataset = DemonstratorProficiencyDataset(config, split=split, logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=False,
        pin_memory=True,
    )
    return val_loader

def load_prediction_files(file_paths):
    """
    加载多个模型的预测结果文件
    :param file_paths: 预测结果文件路径列表
    :return: 包含所有模型预测结果的列表
    """
    all_predictions = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            predictions = json.load(f)
            all_predictions.append(predictions)
    return all_predictions


def ensemble_predictions(all_predictions, weights):
    """
    集成多个模型的预测结果，使用加权平均
    :param all_predictions: 包含所有模型预测结果的列表
    :param weights: 每个模型的权重列表
    :return: 集成后的预测结果
    """

    # 检查所有字典的视频名称列表是否一致
    videos_ref = all_predictions[0]["videos"]
    for pred in all_predictions[1:]:
        if pred["videos"] != videos_ref:
            raise ValueError("所有结果字典的视频名称列表必须一致")

    num_models = len(all_predictions)
    assert len(weights) == num_models, "The length of weights must be equal to the number of models."

    weights = np.array(weights) / sum(weights)

    # 提取所有分数
    all_scores = [np.array(pred["scores"]) for pred in all_predictions]

    # 计算加权平均分数
    weighted_scores = [weights[i] * all_scores[i] for i in range(num_models)]
    ensemble_scores = np.sum(weighted_scores, axis=0).tolist()
  
    # 基于平均分数重新计算预测类别
    ensemble_predictions = [score.index(max(score)) for score in ensemble_scores]

    ensemble_result = {
        "videos": videos_ref,
        "predictions": ensemble_predictions,
        "scores": ensemble_scores,
    }

    return ensemble_result



def ensemble_by_voting(all_predictions):
    """
    借助投票的方式对多个模型的预测结果进行集成
    
    参数:
    all_predictions (list[dict]): 预测结果列表，每个元素都是包含以下内容的字典：
        - videos (list): 视频名称列表
        - predictions (list): 对应的分类预测结果，取值范围为 [0, 3]
    
    返回:
    dict: 集成后的结果，包含：
        - videos (list): 视频名称列表
        - predictions (list): 投票后的分类结果
    """
    # 对输入的有效性进行检查
    if len(all_predictions) < 3:
        raise ValueError("输入的预测结果数量至少为3个")
    
    # 验证所有预测结果中的视频是否一致
    first_videos = all_predictions[0]["videos"]
    for pred in all_predictions[1:]:
        if pred["videos"] != first_videos:
            raise ValueError("所有预测结果中的视频名称列表必须相同")
    
    # 按视频进行投票
    num_videos = len(first_videos)
    final_predictions = []
    
    for i in range(num_videos):
        # 收集所有模型对当前视频的预测结果
        votes = [pred["predictions"][i] for pred in all_predictions]
        
        # 统计每种预测结果出现的次数
        vote_counts = Counter(votes)
        
        # 找出获得票数最多的预测结果
        max_vote = max(vote_counts.values())
        top_predictions = [cls for cls, cnt in vote_counts.items() if cnt == max_vote]
        
        # 若出现平局，则选择编号最小的类别
        final_class = min(top_predictions)
        final_predictions.append(final_class)
    
    return {
        "videos": first_videos,
        "predictions": final_predictions
    }

def main(args):
    """
    主函数，集成预测结果并计算分数
    :param config: 配置文件
    :param prediction_files: 预测结果文件路径列表
    :param weights: 每个模型的权重列表
    """

    print(f"args: {args}")

    if args.test_set:
        prediction_files = test_prediction_files
        weights = test_weights
        split = 'test'
    else:
        prediction_files = val_prediction_files
        weights = val_weights
        split = 'val'

    print("Prediction files: ", prediction_files)
    print("Weights =", weights)

    print(f"Total number of models: {len(prediction_files)}")

    all_predictions = load_prediction_files(prediction_files)

    accuracy_list = []
    result_list = []

    ###### ensemble by scores average ###########
    print("-----------------------start ensemble by score------------------------------------")

    ensemble_result = ensemble_predictions(all_predictions, weights)

    # compute the score
    print("Computing the score")
    config = get_config(args)
    logger, _ = get_logger_and_tb_writer(config, split="val")

    val_loader = get_val_loader(config, split, logger)
    groundtruth = val_loader.dataset.annotations

    accuracy = calculate_accuracy(
        ensemble_result['videos'], 
        ensemble_result['predictions'], 
        groundtruth)
    print(f"Ensemble results by scores - accurancy {accuracy:.4f}")

    accuracy_list.append(accuracy)
    result_list.append(ensemble_result)

    # post processs participant ID by scores 
    post_result_dict = post_process_result_by_participant(ensemble_result, config)
   
    accuracy = calculate_accuracy(
        post_result_dict['videos'], 
        post_result_dict['predictions'], 
        groundtruth)

    print(f'Ensemble accuracy after postprocessing by scores: {accuracy:.4f}')

    # accuracy_list.append(accuracy)
    # result_list.append(post_result_dict)

    # post processs participant ID by voting 
    post_result_dict = post_process_participant_by_voting(ensemble_result, config)
   
    accuracy = calculate_accuracy(
        post_result_dict['videos'], 
        post_result_dict['predictions'], 
        groundtruth)

    print(f'Ensemble accuracy after postprocessing by voting: {accuracy:.4f}')

    # accuracy_list.append(accuracy)
    # result_list.append(post_result_dict)

    ######## ensemble by voting ####################################
    print("-----------------------start ensemble by voting------------------------------------")
    
    voting_result_dict = ensemble_by_voting(all_predictions)
    accuracy = calculate_accuracy(
            voting_result_dict['videos'], 
            voting_result_dict['predictions'], 
            groundtruth)

    print(f"Voting ensemble accurancy {accuracy:.4f}")

    accuracy_list.append(accuracy)
    result_list.append(voting_result_dict)

    # post process #
    post_voting_result = post_process_participant_by_voting(voting_result_dict, config)
    accuracy = calculate_accuracy(
            post_voting_result['videos'], 
            post_voting_result['predictions'], 
            groundtruth)

    print(f"Voting ensemble accurancy after postprocessing {accuracy:.4f}")

    # accuracy_list.append(accuracy)
    # result_list.append(post_voting_result)

    # save the results. 
    print("-----------------------save the result------------------------------------")
    print(accuracy_list)

    # 找到最大准确率对应的索引
    max_index = accuracy_list.index(max(accuracy_list))
    
    # 获取最佳结果
    best_result = result_list[max_index]
    
    # 保存为JSON文件
    result_json_name = f"ensemble_result.json"
    save_dir = "test_output" if args.test_set else "val_output"
    save_path = os.path.join(save_dir, result_json_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(best_result, f, indent=4)
    
    print(f"最佳结果已保存到 {save_path} (准确率: {accuracy_list[max_index]:.4f})")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="config/v1_ego_r3d-18_4frame_d5.yaml",
        help="Config file path of egoexo4D-proficiency-demonstrator",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-proficiency-demonstrator",
    )

    parser.add_argument('--test_set', action='store_true', help='Test set, no need compute score')

    args = parser.parse_args()

    main(args)
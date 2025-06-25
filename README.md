# EgoExo4D_Proficiency_Demonstrator

PCIE_EmoEgo Team's Championship Technical Solution in CVPR2025 EgoExo4D Demonstrator Proficiency Estimation Challenge​
The PCIE_EmoEgo team has secured the championship in the CVPR2025 EgoExo4D Demonstrator Proficiency Estimation Challenge with a groundbreaking technical solution that integrates multi-view fusion, cross-modal learning, and parameter-efficient architecture. Focused on the challenge of evaluating skill proficiency from synchronized egocentric and exocentric videos of skilled human activities, the team's approach leverages the vast multi-modal dataset of EgoExo4D, which spans 1,286 hours of video across 13 cities and captures diverse activities like sports, cooking, and mechanical repair.

### Training
CUDA_VISIBLE_DEVICES=0 python train.py --config_path config/v20_ego_vit-s_6frame_sampling-by-videos_repeat100.yaml

### Validation
CUDA_VISIBLE_DEVICES=1 python val.py \
    --config_path config/v5_ego_r3d-18_1frame_d5.yaml \
    --model_path output/v5_ego_r3d-18_1frame_d5/2025-05-12_01-07-04/v5_best-e1_val-acc-0.47.pt

### Test
CUDA_VISIBLE_DEVICES=1 python test.py \
    --config_path config/v15_ego_vit-s_4frame_sampling-by-videos_repeat100.yaml  \
    --model_path ckpts/v15_best-e2_val-acc-0.48.pt

### Extract images from video
python tools/cut_and_save_image_from_video/cut_and_save_image_from_video.py

### Create annotion pkl file for train, val, test
python tools/create_annotations/create_annotations.py

### Citation
@article{chen2025pcie_pose,
  title={PCIE\_Pose Solution for EgoExo4D Pose and Proficiency Estimation Challenge},
  author={Chen, Feng and Lertniphonphan, Kanokphan and Yan, Qiancheng and Fan, Xiaohui and Xie, Jun and Zhang, Tao and Wang, Zhepeng},
  journal={arXiv preprint arXiv:2505.24411},
  year={2025}
}

# image + IMU fusion
CUDA_VISIBLE_DEVICES=1  python train.py --config_path config/v20_ego_vit-s_6frame_sampling-by-videos_repeat100.yaml

# python dist_train.py --dist --config_path config/v77_egofusion_r3d-18-image_imu_r3d-18-depth-base_mlp-head_8clip_avg-pooling.yaml 


# val

CUDA_VISIBLE_DEVICES=1 python val.py \
    --config_path config/v5_ego_r3d-18_1frame_d5.yaml \
    --model_path output/v5_ego_r3d-18_1frame_d5/2025-05-12_01-07-04/v5_best-e1_val-acc-0.47.pt


# test
CUDA_VISIBLE_DEVICES=1 python test.py \
    --config_path config/v15_ego_vit-s_4frame_sampling-by-videos_repeat100.yaml  \
    --model_path ckpts/v15_best-e2_val-acc-0.48.pt


# ensemble
python ensemble.py --test_set (optional)


# extract images from video
python tools/cut_and_save_image_from_video/cut_and_save_image_from_video.py

# create annotion pkl file for train, val, test
python tools/create_annotations/create_annotations.py

# generate ego body pose, should add ego-bodypose project. 
python tools/generate_ego_body_pose.py

# InternVL 
python predict_by_intervl.py --model_name InternVL3-1B --num_segments 256 --max_num 12 --egoexo exo --new_label
python predict_by_intervl.py --model_name InternVL3-2B --num_segments 32 --max_num 12 --egoexo exo
python predict_by_intervl.py --model_name InternVL3-8B --num_segments 32 --max_num 12 --egoexo exo
python predict_by_intervl.py --model_name InternVL3-8B --num_segments 64 --max_num 12
python predict_by_intervl.py --model_name InternVL3-14B --num_segments 32 --max_num 12
python predict_by_intervl.py --model_name InternVL3-14B --num_segments 64 --max_num 12
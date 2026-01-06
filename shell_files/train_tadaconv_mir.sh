# # Below this is for TadaConvSpatialGPT not that we specifcy in yaml (--cfg) file now instead
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --detector_weights rgb

# # # Below this is for TadaConvSpatialTemporalGPT not that we specifcy in yaml (--cfg) file now instead
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --detector_weights rgb


# MLPspatial Mix
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSpatialmix_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_three_stride_three_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe

# Below is for MLPSpatialTemporalMixer
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmix_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe

# Below is for MLPSpatialTemporalMixerv2
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe
    


# MLPSpatialMixer RGB fullframes
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSpatialmix_same_num_patchesfullframes.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_three_stride_three_fullframes  \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head fullframes

#MLPSpatialTemporalMixer RGB fullframes
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmix_same_num_patchesfullframes.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_fullframes \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head fullframes

# Below is for MLPSpatialTemporalMixerv2
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_diff_num_patchesfullframes_smaller.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_fullframes \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head fullframes \


# MLPspatial Mix blank
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSpatialmix_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_three_stride_three_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head lastframe

# Below is for MLPSpatialTemporalMixer blank
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmix_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head lastframe



# Below is for MLPSpatialTemporalMixerv2 blank
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head lastframe

# MLPSpatialMixer fullframes
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSpatialmix_same_num_patchesfullframes.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_three_stride_three_fullframes  \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head fullframes


#MLPSpatialTemporalMixer RGB fullframes blank
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmix_same_num_patchesfullframes.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_fullframes \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head fullframes

# Below is for MLPSpatialTemporalMixerv2 blank
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patchesfullframes.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_fullframes \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head fullframes


#Randomness Issue
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_small.yaml \
    --batch-size 2 \
    --epochs 3 \
    --lframe 3 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patchesfullframes_without_randomnessv3.yaml \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name randomness_delete_later \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights blank \
    --detection_head fullframes

    # --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patchesfullframes.yaml \



python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/two_detection_heads/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_smaller_2DH.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe_2DH \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe


python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_lastframe_smaller.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe



python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/one_detect_head/strip_mlp/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_stripmlpSTmixv2_same_num_patches_lastframe.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name stripmlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe




python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/one_detect_head/strip_mlp/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_stripmlp_lastframe.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name stripmlp_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe \
    --json_gt_loc ./json_gt/not_santinized_kaist/ \
    --temporal_mosaic \
    --mosaic \




python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_small.yaml \
    --batch-size 8 \
    --epochs 10 \
    --cfg ./models/transformer/one_detect_head/smaller_mlp/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_lastframe_smaller.yaml\
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name delete_later_finding_key_error_kaist \
    --temporal_mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe \
    --json_gt_loc ./json_gt/not_santinized_kaist/ \
    --temporal_mosaic \
    --mosaic \





python train_video.py \
    --data ./data/multispectral_temporal/cvc14_video.yaml \
    --batch-size 8 \
    --epochs 30 \
    --lframe 3 \
    --temporal_stride 3 \
    --cfg ./models/transformer/two_detection_heads/yolov5l_fusion_transformerx3_cvc14_aligned_tadaconv_mlpSTmix_same_num_patches_2DH.yaml \
    --gframe 0 \
    --dataset_used cvc14 \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name yolov5l_fusion_transformerx3_cvc14_aligned_tadaconv_mlpSTmix_same_num_patches_2DH \
    --temporal_mosaic \
    --mosaic \
    --use_tadaconv \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --detector_weights rgb \
    --detection_head lastframe \
    --use_both_labels_for_optimization
    
    # --cfg ./models/transformer/one_detect_head/smaller_mlp/yolov5l_fusion_transformerx3_cvc14_aligned_tadaconv_mlpSTmixv2_same_num_patches_lastframe_smaller.yaml \
    # --json_gt_loc ./json_gt/not_santinized_kaist/ \
=======
    --name fusion_transformerx3_kaist_video_thermal_rgb_backbone_thermal_head_loaded_lframe_three_stride_three \
=======
    --name fusion_transformerx3_kaist_video_thermal_rgb_backbone_rgb_head_loaded_lframe_three_stride_three_mlpSTmixv2_same_num_patches_lastframedeform_smaller_with_grad_clip \
>>>>>>> 728937e (launch/sh files)
    --use_tadaconv \
    --detection_head lastframe \
    --hyp data/hyp.finetune.yaml \
    --use_mode_spec_back_weights \
    --sanitized \
    --detector_weights rgb \
    --temporal_mosaic \
    --mosaic \
    --ignore_high_occ \
    --all_objects \
    --gradient_clip 1 \
    # --save_all_model_epochs \
    # --detection_head default \
    # --hyp data/hyp.scratch_focal_loss.yaml \
    # --save_all_model_epochs \
    # --weights ""
>>>>>>> f230c49 (launch command files)

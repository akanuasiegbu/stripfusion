# Note that use_tadaconv, fusion_stragtegy and detection_head are used to load the correct modules so that attempt_load can load model correctly
# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_lframe_3_stride_3_conf_thres_2 \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialGPT \
#     --detection_head lastframe \
#     --conf-thres 0.2

# --weights ./runs/models_to_keep/fully_tadaconv/tadaconv_fusion_transformerx3_kaist_video_with_logging_nan_relu_used_pretrained_use_one_feature/weights/best.pt \
    # --use_tadaconv \ # would not currently need this because testing but maybe
    # --temporal_mosaic \ # not used during testing


# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_focal2/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_focal2_lframe_3_stride_3_conf_2 \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialGPT \
#     --detection_head lastframe \
#     --conf-thres 0.2


# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three/weights/cur_0.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialGPT \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
    # --multiple_outputs


# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three/weights/cur_0.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialGPT \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_seven_stride_three/weights/cur_0.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_seven_stride_three \
#     --lframe 7 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialGPT \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs




# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three3/weights/cur_0.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three3 \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs    



# echo Finshed Number First************************************************************************************************************************************
 
# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three3/weights/cur_0.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three3 \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs 

# echo Finshed Number Second************************************************************************************************************************************




# python test_video.py \
#     --weights ./runs/models_to_keep/fully_tadaconv/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_backbone_head_loaded2/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name delete_later \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     # --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_three_stride_three_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_three_stride_three_lastframe \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_three_stride_three_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_three_stride_three_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_three_stride_three_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_three_stride_three_lastframe \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_lastframe \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_lastframe/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_lastframe \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head lastframe \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_three_stride_three_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSpatialmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_three_stride_three_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmix_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --multiple_outputs

# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --multiple_outputs


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2

#1
# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2

# #2

# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2

# #4
# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2

# #5
# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2
# #6
# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2

# # 7
# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2

# # 8
# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \


# python  test_video.py \
#     --weights ./runs/train/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --detection_head fullframes \
#     --task test \
#     --exist-ok \
#     --conf-thres 0.2


for i in {0..39}
# do 
#     echo "Looping ... number $i"
# done
do
    python  test_video.py \
    --weights ./runs/train/yolov5l_fusion_transformerx3_cvc14_tadaconv_mlpSTmixv2_same_num_patches_small_2DH/weights/cur_$i.pt \
    --data ./data/multispectral_temporal/cvc14_video_test.yaml \
    --name cvc14_trial_with_only_rgbv2\
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --img-size 640 \
    --save-json \
    --use_tadaconv \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.001 \
    --use_both_labels_for_optimization 
done
# CUDA_VISIBLE_DEVICES='6' \
# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_thermal_rgb_backbone_head_rgb_lframe_three_stride_three/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres02 \
#     --task test \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head fullframes \
#     --exist-ok \
#     --multiple_outputs \
#     --conf-thres 0.2  > logs/test_tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres02.log 2>&1 &


# CUDA_VISIBLE_DEVICES='7' \
# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_thermal_rgb_backbone_head_rgb_lframe_three_stride_three/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres0001 \
#     --task test \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head fullframes \
#     --exist-ok \
#     --multiple_outputs > logs/test_tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres0001.log 2>&1 &
#     # --conf-thres 0.2

# CUDA_VISIBLE_DEVICES='6' \
# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres02 \
#     --task test \
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
#     --exist-ok \
#     --multiple_outputs \
#     --conf-thres 0.2  > logs/test_tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres02.log 2>&1 &


# CUDA_VISIBLE_DEVICES='7' \
# python test_video.py \
#     --weights ./runs/train/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres0001 \
#     --task test \
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
#     --exist-ok \
#     --multiple_outputs > logs/test_tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres0001.log 2>&1 &



# CUDA_VISIBLE_DEVICES='4' \
# python test_video.py \
#     --weights ./runs/train/debug_temporal_mosaic/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name debug_temporal_mosaic_conf_thres02 \
#     --task test \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head fullframes \
#     --exist-ok \
#     --multiple_outputs \
#     --conf-thres 0.2  > logs/test_debug_temporal_mosaic_conf_thres02.log 2>&1 &


# CUDA_VISIBLE_DEVICES='5' \
# python test_video.py \
#     --weights ./runs/train/debug_temporal_mosaic/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name debug_temporal_mosaic_conf_thres0001 \
#     --task test \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head fullframes \
#     --exist-ok \
#     --multiple_outputs > logs/test_debug_temporal_mosaic_conf_thres0001.log 2>&1 &

# CUDA_VISIBLE_DEVICES='4' \
# python test_video.py \
#     --weights ./runs/train/debug_mosaic/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name debug_mosaic_conf_thres02 \
#     --task test \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head fullframes \
#     --exist-ok \
#     --multiple_outputs \
#     --conf-thres 0.2  > logs/test_debug_mosaic_conf_thres02.log 2>&1 &


# CUDA_VISIBLE_DEVICES='5' \
# python test_video.py \
#     --weights ./runs/train/debug_mosaic/weights/best.pt \
#     --data ./data/multispectral_temporal/kaist_video_test.yaml \
#     --name debug_mosaic_conf_thres0001 \
#     --task test \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --regex_search .set...V... \
#     --dataset_used kaist \
#     --img-size 640 \
#     --save-json \
#     --use_tadaconv \
#     --fusion_strategy TadaConvSpatialTemporalGPT \
#     --detection_head fullframes \
#     --exist-ok \
#     --multiple_outputs > logs/test_debug_mosaic_conf_thres0001.log 2>&1 &
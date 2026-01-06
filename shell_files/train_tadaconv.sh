# CUDA_VISIBLE_DEVICES='0,1' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialGPT \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialGPT.log 2>&1 &


# CUDA_VISIBLE_DEVICES='2' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_lframe1_stride1 \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_lframe1_stride1.log 2>&1 &


# CUDA_VISIBLE_DEVICES='0' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_COCO_pretrained_lframe1_stride1 \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --rgb_weights yolov5l.pt \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_COCO_pretrained_lframe1_stride1.log 2>&1 &


# CUDA_VISIBLE_DEVICES='2,3' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_conv \
#     --temporal_mosaic \
#     # --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_conv.log 2>&1 &


# CUDA_VISIBLE_DEVICES='4,5' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_corrected_yaml \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_corrected_yaml.log 2>&1 &


# CUDA_VISIBLE_DEVICES='6,7' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_COCO_pretrained \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --rgb_weights yolov5l.pt \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_COCO_pretrained.log 2>&1 &

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 48 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_bs48 \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_bs48.log 2>&1 &





# CUDA_VISIBLE_DEVICES='2,3' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_midframe.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head midframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe.log 2>&1 &


# CUDA_VISIBLE_DEVICES='4,5' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_midframe.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/bigger_mlp/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_midframe.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_midframe \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head midframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_midframe.log 2>&1 &

# CUDA_VISIBLE_DEVICES='6,7' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_midframe.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_midframe.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_midframe \
#     --temporal_mosaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head midframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_midframe.log 2>&1 &






# CUDA_VISIBLE_DEVICES='0,1' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_lastframe_no_masaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_lastframe_no_masaic.log 2>&1 &


# CUDA_VISIBLE_DEVICES='2,3' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_midframe.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe_no_masaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head midframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe_no_masaic.log 2>&1 &




# CUDA_VISIBLE_DEVICES='4' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe1_stride1.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 1 \
#     --temporal_stride 1 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_lframe1_stride1_lastframe_no_masaic \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_lframe1_stride1_lastframe_no_masaic.log 2>&1 &






# CUDA_VISIBLE_DEVICES='6,7' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     # > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_lastframe.log 2>&1 &


# CUDA_VISIBLE_DEVICES='6,7' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3_midframe.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe_correct \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head midframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_midframe_correct.log 2>&1 &


# CUDA_VISIBLE_DEVICES='3,4' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta.log 2>&1 &



# CUDA_VISIBLE_DEVICES='2,5' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_AddST.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta_AddST \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta_AddST.log 2>&1 &



# CUDA_VISIBLE_DEVICES='3,4' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_ConvGPT_zero_delta \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_ConvGPT_zero_delta.log 2>&1 &



# CUDA_VISIBLE_DEVICES='1,7' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta_cumprod \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta_cumprod.log 2>&1 &

# CUDA_VISIBLE_DEVICES='3' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 2 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta_cumprod \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     # > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta_cumprod.log 2>&1 &

# --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \



# CUDA_VISIBLE_DEVICES='0,1' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/one_detect_head/gpt_varients/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta_2nd_try \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_TadaConvSpatialTemporalGPT_zero_delta_2nd_try.log 2>&1 &


# CUDA_VISIBLE_DEVICES='2,3' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta.log 2>&1 &

# CUDA_VISIBLE_DEVICES='4,5' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta_2nd_try \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta_2nd_try.log 2>&1 &


# CUDA_VISIBLE_DEVICES='6,7' \
# python train_video.py \
#     --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
#     --batch-size 8 \
#     --epochs 40 \
#     --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA.yaml \
#     --lframe 3 \
#     --temporal_stride 3 \
#     --gframe 0 \
#     --dataset_used kaist \
#     --regex_search '.set...V...' \
#     --img-size 640 \
#     --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta_3rd_try \
#     --use_tadaconv \
#     --exist-ok \
#     --detection_head lastframe \
#     --hyp data/hyp.finetune.yaml \
#     --save_all_model_epochs \
#     --use_mode_spec_back_weights \
#     --sanitized \
#     --detector_weights rgb \
#     --temporal_mosaic \
#     --mosaic \
#     > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSA_zero_delta_3rd_try.log 2>&1 &







CUDA_VISIBLE_DEVICES='0,1' \
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
    --batch-size 8 \
    --epochs 40 \
    --cfg ./models/transformer/one_detect_head/bigger_mlp/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_zero_delta \
    --use_tadaconv \
    --exist-ok \
    --detection_head lastframe \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --sanitized \
    --detector_weights rgb \
    --temporal_mosaic \
    --mosaic \
    > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_zero_delta.log 2>&1 &


CUDA_VISIBLE_DEVICES='2,3' \
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
    --batch-size 8 \
    --epochs 40 \
    --cfg ./models/transformer/one_detect_head/bigger_mlp/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_zero_delta_2nd_try \
    --use_tadaconv \
    --exist-ok \
    --detection_head lastframe \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --sanitized \
    --detector_weights rgb \
    --temporal_mosaic \
    --mosaic \
    > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_zero_delta_2nd_try.log 2>&1 &


CUDA_VISIBLE_DEVICES='4,5' \
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
    --batch-size 8 \
    --epochs 40 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_zero_delta_2nd_try \
    --use_tadaconv \
    --exist-ok \
    --detection_head lastframe \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --sanitized \
    --detector_weights rgb \
    --temporal_mosaic \
    --mosaic \
    > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_zero_delta_2nd_try.log 2>&1 &


CUDA_VISIBLE_DEVICES='6,7' \
python train_video.py \
    --data ./data/multispectral_temporal/kaist_video_sanitized_lframe3_stride3.yaml \
    --batch-size 8 \
    --epochs 40 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1.yaml \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --name yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_zero_delta \
    --use_tadaconv \
    --exist-ok \
    --detection_head lastframe \
    --hyp data/hyp.finetune.yaml \
    --save_all_model_epochs \
    --use_mode_spec_back_weights \
    --sanitized \
    --detector_weights rgb \
    --temporal_mosaic \
    --mosaic \
    > logs/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_zero_delta.log 2>&1 &

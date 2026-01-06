# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/tadaconv_fusion_transformerx3_kaist_video_lframe_3_stride_3_pretrained_use_one_feature_temporal_mosaic/best_predictions.json \
#         --name tadaconv_fusion_transformerx3_kaist_video_lframe_3_stride_3_pretrained_use_one_feature_temporal_mosaic \
#         --plot-score



# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/tadaconv_fusion_transformerx3_kaist_video_with_logging_nan_relu_used_pretrained7_lframe_3_stride_3_average_temporal_mosaic/best_predictions.json \
#         --name tadaconv_fusion_transformerx3_kaist_video_with_logging_nan_relu_used_pretrained7_lframe_3_stride_3_average_temporal_mosaic \
#         --plot-score




# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_lframe_3_stride_3/best_predictions.json \
#         --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_lframe_3_stride_3 \
#         --plot-score



# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/fusion_transformerx3_kaist_video_temporal_mosaic_scratch4_lframe_1_stride_1/best_predictions.json \
#         --name fusion_transformerx3_kaist_video_temporal_mosaic_scratch4_lframe_1_stride_1 \
#         --plot-score



# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/fusion_transformerx3_kaist_video_mosaic_scratch3_lframe_1_stride_1/best_predictions.json \
#         --name fusion_transformerx3_kaist_video_mosaic_scratch3_lframe_1_stride_1 \
#         --plot-score


# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_lframe_3_stride_3_conf_thres_2/best_predictions.json \
#         --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_lframe_3_stride_3_conf_thres_2_no_score \
#         # --plot-score


# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_backbone_head_loaded2_3_stride_3_conf_thres_2/best_predictions.json \
#         --name tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_backbone_head_loaded2_3_stride_3_conf_thres_2 \
#         --plot-score


# ""
# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/fusion_transformerx3_kaist_video_focal3_lframe_one_stride_one_conf_2/best_predictions.json \
#         --name fusion_transformerx3_kaist_video_focal3_lframe_one_stride_one_conf_2 \


# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/fusion_transformerx3_kaist_video_scratch_focal2_lframe_one_stride_one_conf_2/best_predictions.json \
#         --name fusion_transformerx3_kaist_video_scratch_focal2_lframe_one_stride_one_conf_2 \


# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/fusion_transformerx3_kaist_video_backbone_head_loaded_lframe_three_stride_three2_conf_2/best_predictions.json \
#         --name fusion_transformerx3_kaist_video_backbone_head_loaded_lframe_three_stride_three2_conf_2 \


# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/fusion_transformerx3_kaist_video_backbone_head_loaded_lframe_one_stride_one2_conf_2/best_predictions.json \
#         --name fusion_transformerx3_kaist_video_backbone_head_loaded_lframe_one_stride_one2_conf_2 \



# python visualization_from_json.py \
#         --data ./data/multispectral_temporal/kaist_video_test.yaml \
#         --dataset_used  kaist \
#         --json_file ./runs/test/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes/best_predictions_ct2.json \
#         --name mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes 



python visualization_from_json.py \
        --data ./data/multispectral_temporal/kaist_video_test.yaml \
        --dataset_used  kaist \
        --json_file ./runs/test/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes/best_predictions_ct2.json \
        --name GroundTruthImages_Thermal

python visualization_from_json.py \
        --data ./data/multispectral_temporal/cvc14_video_test.yaml \
        --dataset_used  cvc14 \
        --json_file ./runs/test/cvc14_trial_with_only_rgbv2/best_predictions_ct2.json \
        --name cvc14_run

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three/best_predictions.json \


# echo Finshed Number 1************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three/best_predictions.json \


# echo Finshed Number 2************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_seven_stride_three/best_predictions.json \

# echo Finshed Number 3************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_seven_stride_three2/best_predictions.json \

# echo Finshed Number 4************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/fusion_transformerx3_kaist_video_thermal_rgb_backbone_thermal_head_loaded_lframe_one_stride_one/best_predictions.json \
# echo Finshed Number 5************************************************************************************************************************************


# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/fusion_transformerx3_kaist_video_thermal_rgb_backbone_rgb_head_loaded_lframe_one_stride_one/best_predictions.json \
# echo Finshed Number 6************************************************************************************************************************************


# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/fusion_transformerx3_kaist_video_thermal_rgb_backbone_rgb_head_loaded_lframe_three_stride_three/best_predictions.json \

# echo Finshed Number 7************************************************************************************************************************************


# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/fusion_transformerx3_kaist_video_thermal_rgb_backbone_thermal_head_loaded_lframe_three_stride_three/best_predictions.json \

# echo Finshed Number 8************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_thermal_lframe_three_stride_three3/best_predictions.json \

# echo Finshed Number 9************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_thermal_rgb_backbone_head_rgb_lframe_three_stride_three3/best_predictions.json \

# echo Finshed Number 10************************************************************************************************************************************

# python ./miss_rate_and_map/map_calc.py \
#     --annFile ./json_gt/kaist_test20_for_map.json \
#     --rstFiles ./runs/test/mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_fullframes/best_predictions_ct001_conf.json \


here="mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_focal_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_focal_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_fullframes
mlpSTmixv2_tadaconv_thermal_rgb_backbone_head_blank_lframe_3_stride_3_drop_25_diff_patches_focal_fullframes
"

for i in $here; do
   python ./miss_rate_and_map/map_calc.py \
    --annFile ./json_gt/kaist_test20_for_map.json \
    --rstFiles ./runs/test/$i/best_predictions_ct001_conf.json
    done

for i in $here; do
   python ./miss_rate_and_map/map_calc.py \
    --annFile ./json_gt/kaist_test20_for_map.json \
    --rstFiles ./runs/test/$i/best_predictions_ct2_conf.json
    done

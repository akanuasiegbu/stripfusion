
python miss_rate_and_map/evaluation_script_cvc14.py \
   --annFile json_gt/cvc-14_test_tl.json \
   --multiple_outputs \
   --rstFiles ./runs/test/yolov5l_fusion_transformerx3_cvc14_tadaconv_mlpSTmixv2_same_num_patches_small_2DH_aligned_data_v2_klbacknl/cur_0_predictions_ct001.json 


python miss_rate_and_map/evaluation_script_cvc14.py \
   --annFile json_gt/cvc-14_test_tl.json \
   --multiple_outputs \
   --rstFiles ./runs/test/cvc14_stripmlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe_datalignev2_klbacknl/cur_0_predictions_ct001.json 

python miss_rate_and_map/evaluation_script_cvc14.py \
   --annFile json_gt/cvc-14_test_tl.json \
   --multiple_outputs \
   --rstFiles ./runs/test/yolov5l_fusion_transformerx3_cvc14_tadaconv_mlpSTmixv2_same_num_patches_small_2DH_aligned_data_v2_resized_with_focal_klbacknl/cur_0_predictions_ct001.json 


python miss_rate_and_map/evaluation_script_cvc14.py \
   --annFile json_gt/cvc-14_test_tl.json \
   --multiple_outputs \
   --rstFiles ./runs/test/cvc14_stripmlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe_dataaligned_resizedv2_klbacknl/cur_0_predictions_ct001.json 


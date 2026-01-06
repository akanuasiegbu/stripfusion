python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --multiple_outputs \
   --rstFiles runs/test/kaist_stripmlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe_wholedata_set_5_klbacknl/cur_0_predictions_ct001.json \

# Not just ouputted for ct2
python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --multiple_outputs \
   --rstFiles runs/test/kaist_stripmlpSTmixv2_tadaconv_thermal_rgb_backbone_head_rgb_lframe_3_stride_3_lastframe_wholedata_set_1_klbacknl/cur_0_predictions_ct001.json \


python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --multiple_outputs \
   --rstFiles runs/test/load_with_rain_with_deform_conv_v2_from_scratch_just_detectorv2/cur_0_predictions_ct001.json \


python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --multiple_outputs \
   --rstFiles runs/test/load_with_rain_with_deform_conv_v2_from_scratch_fusion_detectorv2/cur_0_predictions_ct001.json \


python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --multiple_outputs \
   --rstFiles runs/test/train_with_deform_conv_v2_from_scratchv2/cur_0_predictions_ct001.json \
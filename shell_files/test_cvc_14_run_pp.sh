folder="yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_R_iT_freezeT_fixed_kl_div"

# python miss_rate_and_map/evaluation_script_cvc14.py \
#  --annFile json_gt/cvc-14_test_tl.json \
# --rstFiles  runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.2_rgb_conf_thres_0.2_iou_thresh_0.75_avg.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.2_rgb_conf_thres_0.2_iou_thresh_0.75_max.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.2_rgb_conf_thres_0.2_iou_thresh_0.5_avg.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.2_rgb_conf_thres_0.2_iou_thresh_0.5_max.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.1_rgb_conf_thres_0.1_iou_thresh_0.75_avg.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.1_rgb_conf_thres_0.1_iou_thresh_0.75_max.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.1_rgb_conf_thres_0.1_iou_thresh_0.5_avg.json \
#             runs/test/$folder/cur_39_predictions_ct2_all_pp_fusion_ir_conf_thres_0.1_rgb_conf_thres_0.1_iou_thresh_0.5_max.json




python miss_rate_and_map/evaluation_script_cvc14.py \
 --annFile json_gt/cvc-14_test_tl.json \
--rstFiles  runs/test/$folder/cur_39_predictions_ct2_all_.json \
            runs/test/$folder/cur_39_predictions_ct2_rgb_.json \
            runs/test/$folder/cur_39_predictions_ct2_thermal_.json \

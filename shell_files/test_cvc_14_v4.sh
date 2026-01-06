# runs/train/yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_1_tstride_1_lframe_fixed_kl_div_cvc14_use_only_rgb_anns_verify_again
# runs/train/yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_kl_div_cvc14_use_only_rgb_anns
# runs/train/yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns


# runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst
# runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns
# runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns


python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst \
 --lframe 3 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \



python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst \
 --lframe 3 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_thermal_inference \

python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst \
 --lframe 3 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_thermal_inference \

python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_3_tstride_3_lframe_ablation_why_w_o_tadaconv_worst \
 --lframe 3 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_thermal_inference \
 --pp_fusion_nms \

#######################################################################################################################################################################################################################################

python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \



python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_thermal_inference \


python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_thermal_inference \


python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_thermal_inference \
 --pp_fusion_nms \


#######################################################################################################################################################################################################################################

python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \


python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_thermal_inference \

python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_thermal_inference \


python  test_video.py \
 --weights ./runs/train/yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --name yolov5l_cvc14_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_1_tstride_1_lframe_fixed_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14 \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_thermal_inference \
 --pp_fusion_nms \
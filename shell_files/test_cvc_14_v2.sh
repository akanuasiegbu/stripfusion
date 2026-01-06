# yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_1_tstride_1_lframe_fixed_kl_div_cvc14_use_only_rgb_anns
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div
#################################################################################################################
python test_video.py \
 --weights ./runs/train/yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_1_tstride_1_lframe_fixed_kl_div_cvc14_use_only_rgb_anns/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
 --name percentage_yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_1_tstride_1_lframe_fixed_kl_div_cvc14_use_only_rgb_anns \
 --lframe 1 \
 --temporal_stride 1 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_tadaconv \
 --use_thermal_inference \
 --pp_fusion_nms \
 --kl_cross \
 --use_both_labels_for_optimization




python test_video.py \
 --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_fixed_kl_div/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
 --name percentage_yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_fixed_kl_div \
 --lframe 3 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_tadaconv \
 --use_thermal_inference \
 --pp_fusion_nms \
 --kl_cross \
 --use_both_labels_for_optimization



python test_video.py \
 --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
 --name percentage_yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div \
 --batch-size 32 \
 --lframe 5 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_tadaconv \
 --use_thermal_inference \
 --pp_fusion_nms \
 --kl_cross \
 --use_both_labels_for_optimization


python test_video.py \
 --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
 --name percentage_yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div \
 --batch-size 32 \
 --lframe 7 \
 --temporal_stride 3 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_tadaconv \
 --use_thermal_inference \
 --pp_fusion_nms \
 --kl_cross \
 --use_both_labels_for_optimization


python test_video.py \
 --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt \
 --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
 --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
 --name percentage_yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div \
 --batch-size 32 \
 --lframe 7 \
 --temporal_stride 5 \
 --gframe 0 \
 --regex_search .set...V... \
 --dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
 --img-size 640 \
 --save-json \
 --detection_head lastframe \
 --task test \
 --exist-ok \
 --conf-thres 0.2 \
 --resize_cvc14_for_eval \
 --use_rgb_inference \
 --use_tadaconv \
 --use_thermal_inference \
 --pp_fusion_nms \
 --kl_cross \
 --use_both_labels_for_optimization


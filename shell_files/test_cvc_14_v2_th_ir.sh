python  test_video.py \
--batch-size 32 \
--weights runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy/weights/cur_39.pt \
--data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
--name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy \
--lframe 5 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
--img-size 640 \
--save-json \
--use_tadaconv \
--detection_head lastframe \
--task test \
--exist-ok \
--use_both_labels_for_optimization \
--resize_cvc14_for_eval \
--conf-thres 0.2 \
--use_rgb_inference \
--use_thermal_inference



python  test_video.py \
--batch-size 32 \
--weights runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy/weights/cur_39.pt \
--data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
--name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy \
--lframe 5 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
--img-size 640 \
--save-json \
--use_tadaconv \
--detection_head lastframe \
--task test \
--exist-ok \
--use_both_labels_for_optimization \
--resize_cvc14_for_eval \
--conf-thres 0.2 \
--use_rgb_inference \
--use_thermal_inference \
--pp_fusion_nms

python  test_video.py \
--batch-size 32 \
--weights runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy/weights/cur_39.pt \
--data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
--name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy \
--lframe 5 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
--img-size 640 \
--save-json \
--use_tadaconv \
--detection_head lastframe \
--task test \
--exist-ok \
--use_both_labels_for_optimization \
--resize_cvc14_for_eval \
--conf-thres 0.2 \
--use_thermal_inference



python  test_video.py \
--batch-size 32 \
--weights runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy/weights/cur_39.pt \
--data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
--name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_seed_seventy \
--lframe 5 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
--img-size 640 \
--save-json \
--use_tadaconv \
--detection_head lastframe \
--task test \
--exist-ok \
--use_both_labels_for_optimization \
--resize_cvc14_for_eval \
--conf-thres 0.2 \
--use_rgb_inference




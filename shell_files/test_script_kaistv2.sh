# yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div
##################################################################################################################################################################

python  -m pdb test_video.py  \
   --weights runs/train/yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div \
   --lframe 1 \
   --temporal_stride 1 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist \
   --img-size 640 \
   --save-json \
   --detection_head lastframe \
   --task test \
   --exist-ok \
   --conf-thres 0.2 \
   --use_rgb_inference \
   --use_tadaconv \
   --use_thermal_inference \
   --pp_fusion_nms \
   --kl_cross \
   --use_both_labels_for_optimization

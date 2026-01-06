python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --name yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist \
   --img-size 640 \
   --save-json \
   --use_tadaconv \
   --detection_head lastframe \
   --task test \
   --exist-ok \
   --conf-thres 0.2 \
   --select_thermal_rgb_inference \
   --use_rgb_inference



python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --name yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist \
   --img-size 640 \
   --save-json \
   --use_tadaconv \
   --detection_head lastframe \
   --task test \
   --exist-ok \
   --conf-thres 0.2 \
   --select_thermal_rgb_inference \
   --use_thermal_inference



   python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --name yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist \
   --img-size 640 \
   --save-json \
   --use_tadaconv \
   --detection_head lastframe \
   --task test \
   --exist-ok \
   --conf-thres 0.2 \
   --select_thermal_rgb_inference \
   --use_rgb_inference



python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --name yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist \
   --img-size 640 \
   --save-json \
   --use_tadaconv \
   --detection_head lastframe \
   --task test \
   --exist-ok \
   --conf-thres 0.2 \
   --select_thermal_rgb_inference \
   --use_thermal_inference

python  test_video.py \
--weights runs/train/yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T/weights/cur_24.pt \
--data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
--name yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T \
--lframe 3 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used kaist \
--img-size 640 \
--save-json \
--detection_head lastframe \
--task test \
--exist-ok \
--conf-thres 0.2 \
--select_thermal_rgb_inference \
--use_rgb_inference \
--multiple_outputs
# --use_tadaconv \


python  test_video.py \
--weights runs/train/yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T/weights/cur_24.pt \
--data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
--name yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T \
--lframe 3 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used kaist \
--img-size 640 \
--save-json \
--detection_head lastframe \
--task test \
--exist-ok \
--conf-thres 0.2 \
--select_thermal_rgb_inference \
--use_thermal_inference \
--multiple_outputs
# --use_tadaconv \


python  test_video.py \
--weights runs/train/yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T/weights/cur_24.pt \
--data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
--name yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T \
--lframe 3 \
--temporal_stride 3 \
--gframe 0 \
--regex_search .set...V... \
--dataset_used kaist \
--img-size 640 \
--save-json \
--detection_head lastframe \
--task test \
--exist-ok \
--conf-thres 0.2 \
--select_thermal_rgb_inference \
--multiple_outputs
# --use_tadaconv \
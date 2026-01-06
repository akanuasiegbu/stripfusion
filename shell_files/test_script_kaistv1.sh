# yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div
# runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T
# runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_5_lframe
# yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_7_lframe
# yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_temporal_stride_7_lframe
##################################################################################################################################################################

python test_video.py \
   --weights runs/train/yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div \
   --lframe 1 \
   --temporal_stride 1 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization


python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization



python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_5_lframe/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_5_lframe \
   --batch-size 32 \
   --lframe 5 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization




python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_7_lframe/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_7_lframe \
   --batch-size 32 \
   --lframe 7 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization


python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_temporal_stride_7_lframe/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_temporal_stride_7_lframe \
   --batch-size 32 \
   --lframe 7 \
   --temporal_stride 10 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization



# yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div
# yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div
# yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_5_lframe_fixed_kl_div
# yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_7_lframe_fixed_kl_div
# yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_tstride_7_lframe_fixed_kl_div
##################################################################################################################################################################

python test_video.py \
   --weights runs/train/yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_1_tstride_1_lframe_fixed_kl_div \
   --lframe 1 \
   --temporal_stride 1 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization


python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div \
   --lframe 3 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization



python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_5_lframe_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_5_lframe_fixed_kl_div \
   --batch-size 32 \
   --lframe 5 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization




python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_7_lframe_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_7_lframe_fixed_kl_div \
   --batch-size 32 \
   --lframe 7 \
   --temporal_stride 3 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization


python  test_video.py \
   --weights runs/train/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_tstride_7_lframe_fixed_kl_div/weights/cur_24.pt \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
   --name percentage_yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_tstride_7_lframe_fixed_kl_div \
   --batch-size 32 \
   --lframe 7 \
   --temporal_stride 10 \
   --gframe 0 \
   --regex_search .set...V... \
   --dataset_used kaist_reliability_test \
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
   --use_both_labels_for_optimization
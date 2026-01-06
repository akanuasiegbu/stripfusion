#Table VIII
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze_backbone_init_bb_both_heads_iR_R_iT_T_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_fixed_kl_div


        rgb_c="0.2 0.1"
    iou_thresh="0.75 0.5"
##############################################################################################################################
for rgb in $rgb_c; do
  for iou in $iou_thresh; do
    python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --use_tadaconv \
    --img-size 640 \
    --save-json \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.2 \
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --use_rgb_inference \
    --use_thermal_inference \
    --pp_fusion_nms \
    --thermal_conf_thresh $rgb \
    --rgb_conf_thresh $rgb \
    --iou_thresh $iou
  done
done


for rgb in $rgb_c; do
  for iou in $iou_thresh; do
    python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --use_tadaconv \
    --img-size 640 \
    --save-json \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.2 \
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --use_rgb_inference \
    --use_thermal_inference \
    --pp_fusion_nms \
    --thermal_conf_thresh $rgb \
    --rgb_conf_thresh $rgb \
    --iou_thresh $iou \
    --use_max_conf
  done
done


##############################################################################################################################
for rgb in $rgb_c; do
  for iou in $iou_thresh; do
    python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze_backbone_init_bb_both_heads_iR_R_iT_T_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze_backbone_init_bb_both_heads_iR_R_iT_T_fixed_kl_div \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --use_tadaconv \
    --img-size 640 \
    --save-json \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.2 \
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --use_rgb_inference \
    --use_thermal_inference \
    --pp_fusion_nms \
    --thermal_conf_thresh $rgb \
    --rgb_conf_thresh $rgb \
    --iou_thresh $iou
  done
done


for rgb in $rgb_c; do
  for iou in $iou_thresh; do
    python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze_backbone_init_bb_both_heads_iR_R_iT_T_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze_backbone_init_bb_both_heads_iR_R_iT_T_fixed_kl_div \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --use_tadaconv \
    --img-size 640 \
    --save-json \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.2 \
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --use_rgb_inference \
    --use_thermal_inference \
    --pp_fusion_nms \
    --thermal_conf_thresh $rgb \
    --rgb_conf_thresh $rgb \
    --iou_thresh $iou \
    --use_max_conf
  done
done


##############################################################################################################################
for rgb in $rgb_c; do
  for iou in $iou_thresh; do
    python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_fixed_kl_div \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --use_tadaconv \
    --img-size 640 \
    --save-json \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.2 \
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --use_rgb_inference \
    --use_thermal_inference \
    --pp_fusion_nms \
    --thermal_conf_thresh $rgb \
    --rgb_conf_thresh $rgb \
    --iou_thresh $iou
  done
done


for rgb in $rgb_c; do
  for iou in $iou_thresh; do
    python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T_fixed_kl_div \
    --lframe 3 \
    --temporal_stride 3 \
    --gframe 0 \
    --regex_search .set...V... \
    --dataset_used cvc14 \
    --use_tadaconv \
    --img-size 640 \
    --save-json \
    --detection_head lastframe \
    --task test \
    --exist-ok \
    --conf-thres 0.2 \
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --use_rgb_inference \
    --use_thermal_inference \
    --pp_fusion_nms \
    --thermal_conf_thresh $rgb \
    --rgb_conf_thresh $rgb \
    --iou_thresh $iou \
    --use_max_conf
  done
done



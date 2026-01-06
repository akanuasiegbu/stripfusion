# Table VI with KL
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div

        rgb_c="0.2 0.1"
    iou_thresh="0.75 0.5"
##############################################################################################################################

python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div \
    --lframe 5 \
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
 

python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div \
    --lframe 5 \
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
  

python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div \
    --lframe 5 \
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
    --use_thermal_inference \



##############################################################################################################################
python  test_video.py \
  --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt  \
  --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
  --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div \
  --lframe 7 \
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
 
python  test_video.py \
  --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt  \
  --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
  --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div \
  --lframe 7 \
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
  --use_thermal_inference \


 python  test_video.py \
  --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt  \
  --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
  --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div \
  --lframe 7 \
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
 




##############################################################################################################################

python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div \
    --lframe 7 \
    --temporal_stride 5 \
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

python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div \
    --lframe 7 \
    --temporal_stride 5 \
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
    --use_thermal_inference \

python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div/weights/cur_39.pt  \
    --data ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div \
    --lframe 7 \
    --temporal_stride 5 \
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

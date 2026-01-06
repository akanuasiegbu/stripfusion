# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T
###############################################################################################################
python  test_video.py \
    --weights ./runs/train/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/weights/cur_39.pt \
    --data  ./data/multispectral_temporal/cvc14_video_test_resizedv2.yaml  \
    --name yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T  \
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
    --use_both_labels_for_optimization \
    --resize_cvc14_for_eval \
    --select_thermal_rgb_inference \
    --use_thermal_inference \
    --use_tadaconv \
    --multiple_outputs





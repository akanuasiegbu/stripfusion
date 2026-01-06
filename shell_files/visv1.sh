python vis_where_models_missed.py \
    --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
    --dataset_used  kaist \
    --rstFiles runs/test/yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/cur_24_predictions_ct2_all.json \
    --name yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
    --plot_detections \
    --plot_gt_on_top \
    --exist-ok
    # --plot_rgb_image \


python vis_where_models_missed.py \
    --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
    --dataset_used  kaist \
    --rstFiles runs/test/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/cur_24_predictions_ct2_all.json \
    --name yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
    --plot_detections \
    --plot_gt_on_top \
    --exist-ok
    # --plot_rgb_image \


python vis_where_models_missed.py \
    --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
    --dataset_used  kaist \
    --rstFiles runs/test/yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/cur_24_predictions_ct2_all.json \
    --name yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
    --plot_detections \
    --plot_gt_on_top \
    --exist-ok
    # --plot_rgb_image \



python vis_where_models_missed.py \
    --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
    --dataset_used  kaist \
    --rstFiles runs/test/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/cur_24_predictions_ct2_all.json \
    --name yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T \
    --plot_detections \
    --plot_gt_on_top \
    --exist-ok
    # --plot_rgb_image \


python train_video.py \
--data ./data/multispectral_temporal/cvc14_video_aligned_resizedv2_whole.yaml \
--batch-size 6 \
--epochs 40 \
--cfg ./models/transformer/two_detection_heads/ind_fusion_heads/yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_KLBackNL_full.yaml \
--lframe 1 \
--temporal_stride 1 \
--gframe 0 \
--dataset_used cvc14_align_resizedv2_use_only_rgb_anns \
--regex_search '.set...V...' \
--img-size 640 \
--name yolov5l_cvc14_with_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_1_tstride_1_lframe_fixed_kl_div_cvc14_use_only_rgb_anns  \
--temporal_mosaic \
--mosaic \
--hyp data/hyp.finetune_focal_loss_high_obj_low_scale.yaml \
--save_all_model_epochs \
--resize_cvc14_for_eval \
--detection_head lastframe \
--use_both_labels_for_optimization \
--detector_weights both \
--use_mode_spec_back_weights \
--thermal_weights yolov5l_kaist_best_thermal.pt \
--rgb_weights yolov5l_kaist_best_rgb.pt \
--freeze_model \
--freeze_bb_rgb_bb_ir_det_ir \
--use_tadaconv



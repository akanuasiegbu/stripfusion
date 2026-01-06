files="yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_2_freeze3_init_bb_both_heads_iR_freezeR_iT_T_fixed_kl_div
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_R_iT_freezeT_fixed_kl_div
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze_backbone_init_bb_both_heads_iR_R_iT_T_fixed_kl_div
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_no_freezing_init_bb_both_heads_iR_R_iT_T_fixed_kl_div
"

look_here="/home/akanu/akanu/projects/fusion_therm_rgb/state-of-art-detectors/multispectral-video-object-detection/runs/train/"

mk_fold='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_kaist_model_weights/lframe_3_tstride_3/'

weights="/weights/cur_24.pt"


# make folders
# for i in $files; do
#     mkdir "$mk_fold$i"
#     done



# move weights
# for folder_p in $files; do
#     cp -r "$look_here$folder_p$weights"    "$mk_fold$folder_p"
#     echo "##########################"
#     done


look_here_test="/home/akanu/akanu/projects/fusion_therm_rgb/state-of-art-detectors/multispectral-video-object-detection/runs/test/"

json_files="/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/json_files/fixed_kl_div/"

j_all="/cur_24_predictions_ct2_conf_all.json"
j_rgb="/cur_24_predictions_ct2_conf_rgb.json"
j_thermal="/cur_24_predictions_ct2_conf_thermal.json"

for folder_p in $files; do
    # mkdir "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
    echo "##########################"
    done



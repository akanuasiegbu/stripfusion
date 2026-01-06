files_3_5="
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_5_lframe
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_5_lframe_fixed_kl_div
"

files_3_7="
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_temporal_stride_7_lframe
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T_3_tstride_7_lframe_fixed_kl_div
"

files_10_7="
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_temporal_stride_7_lframe
yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_point5_freeze3_init_bb_both_heads_iR_freezeR_iT_T_10_tstride_7_lframe_fixed_kl_div
"

look_here="/home/akanu/akanu/projects/fusion_therm_rgb/state-of-art-detectors/multispectral-video-object-detection/runs/train/"

mk_fold_3_5='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_kaist_model_weights/lframe_5_tstride_3/'
mk_fold_3_7='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_kaist_model_weights/lframe_7_tstride_3/'
mk_fold_10_7='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_kaist_model_weights/lframe_7_tstride_10/'

weights="/weights/cur_24.pt"


# make folders

# for i in $files_3_5; do
#     mkdir "$mk_fold_3_5$i"
#     done

# for i in $files_3_7; do
#     mkdir "$mk_fold_3_7$i"
#     done

# for i in $files_10_7; do
#     mkdir "$mk_fold_10_7$i"
#     done






# move weights
# for folder_p in $files_3_5; do
#     # echo -r "$look_here$folder_p$weights"    "$mk_fold$folder_p"
#     cp -r "$look_here$folder_p$weights"    "$mk_fold_3_5$folder_p"
#     echo "##########################"
#     done

# for folder_p in $files_3_7; do
#     # echo -r "$look_here$folder_p$weights"    "$mk_fold$folder_p"
#     cp -r "$look_here$folder_p$weights"    "$mk_fold_3_7$folder_p"
#     echo "##########################"
#     done

# for folder_p in $files_10_7; do
#     # echo -r "$look_here$folder_p$weights"    "$mk_fold$folder_p"
#     cp -r "$look_here$folder_p$weights"    "$mk_fold_10_7$folder_p"
#     echo "##########################"
#     done










look_here_test="/home/akanu/akanu/projects/fusion_therm_rgb/state-of-art-detectors/multispectral-video-object-detection/runs/test/"

json_files="/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/json_files/fixed_kl_div/"

j_all="/cur_24_predictions_ct2_conf_all.json"
j_rgb="/cur_24_predictions_ct2_conf_rgb.json"
j_thermal="/cur_24_predictions_ct2_conf_thermal.json"


for folder_p in $files_3_5; do
    mkdir "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
    echo "##########################"
    done

for folder_p in $files_3_7; do
    mkdir "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
    echo "##########################"
    done

for folder_p in $files_10_7; do
    mkdir "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
    cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
    echo "##########################"
    done



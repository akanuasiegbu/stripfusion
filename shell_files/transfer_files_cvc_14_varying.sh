here1="0 1 2 3"
here2="0.2 0.3 0.3"

for i in $here1; do
  for j in $here2; do
    echo "$i $j"
  done
done


# # files="
# files_5_7="yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_5_tstride_7_lframe_fixed_kl_div"
# files_3_5="yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_5_lframe_fixed_kl_div"
# files_3_7="yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_without_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe
# yolov5l_cvc14_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_R_iT_freezeT_3_tstride_7_lframe_fixed_kl_div
# "

# look_here="/home/akanu/akanu/projects/fusion_therm_rgb/state-of-art-detectors/multispectral-video-object-detection/runs/train/"

# mk_fold_5_7='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_cvc14_model_weights/lframe_7_tstride_5/'
# mk_fold_3_5='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_cvc14_model_weights/lframe_5_tstride_3/'
# mk_fold_3_7='/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/new_cvc14_model_weights/lframe_7_tstride_3/'

# weights="/weights/cur_39.pt"


# make folders
# for i in $files_5_7; do
#     mkdir "$mk_fold_5_7$i"
#     done

# for i in $files_3_5; do
#     mkdir "$mk_fold_3_5$i"
#     done

# for i in $files_3_7; do
#     mkdir "$mk_fold_3_7$i"
#     done




# move weights
# for folder_p in $files_5_7; do
#     # echo -r "$look_here$folder_p$weights"    "$mk_fold$folder_p"
#     cp -r "$look_here$folder_p$weights"    "$mk_fold_5_7$folder_p"
#     echo "##########################"
#     done


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







# look_here_test="/home/akanu/akanu/projects/fusion_therm_rgb/state-of-art-detectors/multispectral-video-object-detection/runs/test/"

# json_files="/mnt/ws-frb/projects/thermal_rgb/files_miracle_sharing_with_nitin/models_to_check/json_files/fixed_kl_div/"

# j_all="/cur_39_predictions_ct2_conf_all.json"
# j_rgb="/cur_39_predictions_ct2_conf_rgb.json"
# j_thermal="/cur_39_predictions_ct2_conf_thermal.json"

# for folder_p in $files_5_7; do
#     # mkdir "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
#     # echo "##########################"
#     done


# for folder_p in $files_3_5; do
#     # mkdir "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
#     # echo "##########################"
#     done


# for folder_p in $files_3_7; do
#     # mkdir "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_all"    "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_rgb"    "$json_files$folder_p"
#     cp -r "$look_here_test$folder_p$j_thermal"    "$json_files$folder_p"
#     # echo "##########################"
#     done


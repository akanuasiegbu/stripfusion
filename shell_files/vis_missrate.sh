python visualization_using_missrate.py \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --dataset_used  kaist \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles runs/state_of_arts/MBNet_result.txt  \
   --name heavy_occlusion_kaist_all \
   --plot_detections \
   --plot_gt_on_top \
   --exist-ok \
   --filter_occlusion
   # --plot_rgb_image \


python visualization_using_missrate.py \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --dataset_used  kaist \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles runs/state_of_arts/ARCNN_result.txt  \
   --name heavy_occlusion_kaist_all \
   --plot_detections \
   --plot_gt_on_top \
   --exist-ok \
   --filter_occlusion
   # --plot_rgb_image \



python visualization_using_missrate.py \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --dataset_used  kaist \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles runs/state_of_arts/MLPD_result.txt  \
   --name heavy_occlusion_kaist_all \
   --plot_detections \
   --plot_gt_on_top \
   --exist-ok \
   --filter_occlusion
   # --plot_rgb_image \



python visualization_using_missrate.py \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --dataset_used  kaist \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles runs/state_of_arts/MS-DETR_result.txt  \
   --name heavy_occlusion_kaist_all \
   --plot_detections \
   --plot_gt_on_top \
   --exist-ok \
   --filter_occlusion
   # --plot_rgb_image \




python visualization_using_missrate.py \
   --data ./data/multispectral_temporal/kaist_video_test_nc_1.yaml \
   --dataset_used  kaist \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles ./runs/test/yolov5l_kaist_tadaconv_stripmlpSTmixv2_lastframe_2DH_with_kl_div_freeze3_init_bb_both_heads_iR_freezeR_iT_T/cur_24_predictions_ct2_all.json \
   --name heavy_occlusion_kaist_all \
   --plot_detections \
   --plot_gt_on_top \
   --exist-ok \
   --filter_occlusion
   # --plot_rgb_image \


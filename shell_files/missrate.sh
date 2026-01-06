
njotwani_kaist_withkldiv_sept122
python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles /home/nitin/test/njotwani_kaist_withkldiv_sept122/cur_24_predictions_ct2_all.json


python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFiles /home/nitin/test/njotwani_kaist_withoutkldiv_sept10/cur_24_predictions_ct2_all.json


python miss_rate_and_map/evaluation_script.py \
   --annFile miss_rate_and_map/KAIST_annotation.json \
   --rstFile ./runs/test/yolov5l_kaist_without_tadaconv_stripmlpSTmixv2_lastframe_2DH_no_kl_div_no_freezing_init_bb_both_heads_iR_R_iT_T/cur_24_predictions_ct001_all.json

python miss_rate_and_map/evaluation_script.py \
    --annFile miss_rate_and_map/KAIST_annotation.json \
    --rstFiles runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres02 \
    --multiple_outputs \
    --evalFig runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres02/eval_fig.jpg \
    > logs/evaluation_script/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres02.log 2>&1 &

python miss_rate_and_map/evaluation_script.py \
    --annFile miss_rate_and_map/KAIST_annotation.json \
    --rstFiles runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres02 \
    --multiple_outputs \
    --evalFig runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres02/eval_fig.jpg \
    > logs/evaluation_script/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres02.log 2>&1 &

python miss_rate_and_map/evaluation_script.py \
    --annFile miss_rate_and_map/KAIST_annotation.json \
    --rstFiles runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres0001 \
    --multiple_outputs \
    --evalFig runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres0001/eval_fig.jpg \
    > logs/evaluation_script/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_fullframes_focal2_lframe_3_stride_3_conf_thres0001.log 2>&1 &

python miss_rate_and_map/evaluation_script.py \
    --annFile miss_rate_and_map/KAIST_annotation.json \
    --rstFiles runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres0001 \
    --multiple_outputs \
    --evalFig runs/test/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres0001/eval_fig.jpg \
    > logs/evaluation_script/tadaconv_fusion_transformerx3_kaist_video_TadaConvSpatialTemporalGPT_lastframe_focal2_lframe_3_stride_3_conf_thres0001.log 2>&1 &

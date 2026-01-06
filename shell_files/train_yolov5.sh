python train_video.py \
    --data ./data/multispectral_temporal/kaist_video.yaml \
    --batch-size 8 \
    --epochs 80 \
    --cfg ./models/transformer/yolov5l_fusion_transformerx3_Kaist_aligned.yaml \
    --name fusion_transformerx3_kaist_video_focal\
    --lframe 1 \
    --temporal_stride 1 \
    --gframe 0 \
    --dataset_used kaist \
    --regex_search '.set...V...' \
    --img-size 640 \
    --temporal_mosaic \
    --fusion_strategy GPT \
    --detection_head default \
    --hyp data/hyp.scratch_focal_loss.yaml \
    --save_all_model_epochs \
    # --weights ""